#pragma once

#include <QObject>
#include <QProcess>
#include <QString>
#include <QStringList>

class WorkspaceBridge final : public QObject {
    Q_OBJECT
    Q_PROPERTY(QString workspace READ workspace NOTIFY workspaceChanged)
    Q_PROPERTY(QString status READ status NOTIFY statusChanged)
    Q_PROPERTY(QString serverUrl READ serverUrl NOTIFY serverChanged)
    Q_PROPERTY(bool serverRunning READ serverRunning NOTIFY serverChanged)
    Q_PROPERTY(QString log READ log NOTIFY logChanged)
    Q_PROPERTY(QStringList recentWorkspaces READ recentWorkspaces NOTIFY recentWorkspacesChanged)

public:
    explicit WorkspaceBridge(QString workspace, QObject* parent = nullptr);
    ~WorkspaceBridge() override;

    QString workspace() const { return workspace_; }
    QString status() const { return status_; }
    QString serverUrl() const { return server_url_; }
    bool serverRunning() const { return process_.state() != QProcess::NotRunning; }
    QString log() const { return log_; }
    QStringList recentWorkspaces() const { return recent_workspaces_; }

    Q_INVOKABLE bool selectWorkspace(const QString& path);
    Q_INVOKABLE bool initializeWorkspace();
    Q_INVOKABLE void startServer();
    Q_INVOKABLE void stopServer();
    Q_INVOKABLE void openWorkbench();

signals:
    void workspaceChanged();
    void statusChanged();
    void serverChanged();
    void logChanged();
    void recentWorkspacesChanged();

private:
    bool ensureWorkspace(QString* error = nullptr);
    void setStatus(const QString& value);
    void rememberWorkspace();
    void appendLog(const QString& value);

    QString workspace_;
    QString status_;
    QString server_url_;
    QString log_;
    QStringList recent_workspaces_;
    QProcess process_;
};
