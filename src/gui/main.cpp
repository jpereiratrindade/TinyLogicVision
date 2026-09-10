#include "workspace_bridge.hpp"

#include <QCommandLineOption>
#include <QCommandLineParser>
#include <QCoreApplication>
#include <QDir>
#include <QGuiApplication>
#include <QQmlApplicationEngine>
#include <QQmlContext>
#include <QTimer>

int main(int argc, char** argv) {
    QGuiApplication application(argc, argv);
    QCoreApplication::setApplicationName("tinyvision-gui");
    QCoreApplication::setOrganizationName("TinyLogicVision");
    QCoreApplication::setApplicationVersion(TINYVISION_VERSION);

    QCommandLineParser parser;
    parser.addHelpOption();
    parser.addVersionOption();
    const QCommandLineOption workspace({"w", "workspace"}, "TinyLogicVision workspace", "path",
        QStringLiteral(TINYVISION_SOURCE_DIR) + "/.tinyvision");
    const QCommandLineOption smoke("smoke", "Exit after GUI startup validation");
    const QCommandLineOption exerciseServer("exercise-server", "Start and validate the local workspace server");
    parser.addOption(workspace);
    parser.addOption(smoke);
    parser.addOption(exerciseServer);
    parser.process(application);

    WorkspaceBridge bridge(parser.value(workspace));
    QQmlApplicationEngine engine;
    engine.rootContext()->setContextProperty("bridge", &bridge);
    engine.loadFromModule("TinyLogicVision", "Main");
    if (engine.rootObjects().isEmpty()) return 1;
    if (parser.isSet(exerciseServer)) {
        bridge.startServer();
        QTimer::singleShot(1200, &application, [&application, &bridge] {
            application.exit(bridge.serverRunning() ? 0 : 2);
        });
    } else if (parser.isSet(smoke)) {
        QTimer::singleShot(350, &application, &QCoreApplication::quit);
    }
    return application.exec();
}
