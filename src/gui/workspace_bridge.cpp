#include "workspace_bridge.hpp"

#include <QCoreApplication>
#include <QDateTime>
#include <QDesktopServices>
#include <QDir>
#include <QFileInfo>
#include <QJsonDocument>
#include <QJsonObject>
#include <QProcessEnvironment>
#include <QSaveFile>
#include <QSettings>
#include <QTcpServer>
#include <QUrl>

namespace {
QString normalizedPath(const QString& value) {
    return QDir::cleanPath(QFileInfo(value).absoluteFilePath());
}
}

WorkspaceBridge::WorkspaceBridge(QString workspace, QObject* parent)
    : QObject(parent), workspace_(normalizedPath(std::move(workspace))) {
    QSettings settings;
    recent_workspaces_ = settings.value("recentWorkspaces").toStringList();

    connect(&process_, &QProcess::readyReadStandardOutput, this, [this] {
        appendLog(QString::fromUtf8(process_.readAllStandardOutput()));
    });
    connect(&process_, &QProcess::readyReadStandardError, this, [this] {
        appendLog(QString::fromUtf8(process_.readAllStandardError()));
    });
    connect(&process_, &QProcess::started, this, [this] {
        setStatus("Servidor local ativo");
        emit serverChanged();
    });
    connect(&process_, &QProcess::errorOccurred, this, [this](QProcess::ProcessError) {
        setStatus("Falha ao iniciar o servidor: " + process_.errorString());
        emit serverChanged();
    });
    connect(&process_, qOverload<int, QProcess::ExitStatus>(&QProcess::finished), this,
            [this](int exitCode, QProcess::ExitStatus) {
        setStatus(QString("Servidor encerrado (código %1)").arg(exitCode));
        emit serverChanged();
    });

    initializeWorkspace();
}

WorkspaceBridge::~WorkspaceBridge() {
    stopServer();
}

void WorkspaceBridge::setStatus(const QString& value) {
    status_ = value;
    emit statusChanged();
}

void WorkspaceBridge::appendLog(const QString& value) {
    log_ += value;
    constexpr qsizetype maxLog = 24000;
    if (log_.size() > maxLog) log_ = log_.right(maxLog);
    emit logChanged();
}

void WorkspaceBridge::rememberWorkspace() {
    recent_workspaces_.removeAll(workspace_);
    recent_workspaces_.prepend(workspace_);
    while (recent_workspaces_.size() > 8) recent_workspaces_.removeLast();
    QSettings settings;
    settings.setValue("recentWorkspaces", recent_workspaces_);
    emit recentWorkspacesChanged();
}

bool WorkspaceBridge::ensureWorkspace(QString* error) {
    QDir root(workspace_);
    if (!root.exists() && !QDir().mkpath(workspace_)) {
        if (error) *error = "Não foi possível criar o diretório";
        return false;
    }
    if (!QFileInfo(workspace_).isDir()) {
        if (error) *error = "O caminho não é um diretório";
        return false;
    }
    for (const auto& name : {"uploads", "datasets", "models", "runs", "manifests"}) {
        if (!root.mkpath(name)) {
            if (error) *error = QString("Não foi possível criar %1").arg(name);
            return false;
        }
    }

    const QString manifestPath = root.filePath("workspace.json");
    if (QFileInfo::exists(manifestPath)) {
        QFile manifestFile(manifestPath);
        if (!manifestFile.open(QIODevice::ReadOnly)) {
            if (error) *error = "Não foi possível ler workspace.json";
            return false;
        }
        const auto document = QJsonDocument::fromJson(manifestFile.readAll());
        if (!document.isObject() || document.object().value("schema").toString() != "tinylogicvision.workspace/v1") {
            if (error) *error = "workspace.json possui schema incompatível";
            return false;
        }
    } else {
        QJsonObject manifest{
            {"schema", "tinylogicvision.workspace/v1"},
            {"name", QFileInfo(workspace_).fileName()},
            {"created_at", QDateTime::currentDateTimeUtc().toString(Qt::ISODateWithMs)},
        };
        QSaveFile output(manifestPath);
        if (!output.open(QIODevice::WriteOnly) ||
            output.write(QJsonDocument(manifest).toJson(QJsonDocument::Indented)) < 0 || !output.commit()) {
            if (error) *error = "Não foi possível gravar workspace.json";
            return false;
        }
    }
    return true;
}

bool WorkspaceBridge::selectWorkspace(const QString& path) {
    if (serverRunning()) {
        setStatus("Encerre o servidor antes de trocar de workspace");
        return false;
    }
    const QString candidate = normalizedPath(QUrl(path).isLocalFile() ? QUrl(path).toLocalFile() : path);
    if (candidate.isEmpty()) return false;
    workspace_ = candidate;
    emit workspaceChanged();
    return initializeWorkspace();
}

bool WorkspaceBridge::initializeWorkspace() {
    QString error;
    if (!ensureWorkspace(&error)) {
        setStatus("Workspace inválido: " + error);
        return false;
    }
    rememberWorkspace();
    setStatus("Workspace pronto");
    return true;
}

void WorkspaceBridge::startServer() {
    if (serverRunning()) {
        setStatus("O servidor já está ativo");
        return;
    }
    if (!initializeWorkspace()) return;

    QTcpServer probe;
    quint16 port = 8080;
    if (!probe.listen(QHostAddress::LocalHost, port)) {
        if (!probe.listen(QHostAddress::LocalHost, 0)) {
            setStatus("Não foi possível reservar uma porta local");
            return;
        }
        port = probe.serverPort();
    }
    probe.close();

    server_url_ = QString("http://127.0.0.1:%1/").arg(port);
    log_.clear();
    emit logChanged();
    QProcessEnvironment environment = QProcessEnvironment::systemEnvironment();
    environment.insert("PYTHONUNBUFFERED", "1");
    process_.setProcessEnvironment(environment);
    process_.setProgram("python3");
    process_.setArguments({QStringLiteral(TINYVISION_SOURCE_DIR) + "/web/server.py",
                           "--workspace", workspace_, "--port", QString::number(port), "--no-browser"});
    process_.setWorkingDirectory(QStringLiteral(TINYVISION_SOURCE_DIR));
    setStatus("Iniciando servidor local…");
    emit serverChanged();
    process_.start();
}

void WorkspaceBridge::stopServer() {
    if (!serverRunning()) return;
    process_.terminate();
    if (!process_.waitForFinished(2000)) {
        process_.kill();
        process_.waitForFinished(1000);
    }
}

void WorkspaceBridge::openWorkbench() {
    if (!serverRunning()) {
        setStatus("Inicie o servidor antes de abrir o workbench");
        return;
    }
    QDesktopServices::openUrl(QUrl(server_url_));
}
