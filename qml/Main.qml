import QtQuick
import QtQuick.Controls
import QtQuick.Layouts
import QtQuick.Dialogs

ApplicationWindow {
    id: window
    width: 920
    height: 620
    minimumWidth: 720
    minimumHeight: 520
    visible: true
    title: "TinyLogicVision — Workspace"
    color: "#08101d"

    palette.window: "#08101d"
    palette.windowText: "#e5edf8"
    palette.base: "#0c1626"
    palette.text: "#e5edf8"
    palette.button: "#17243a"
    palette.buttonText: "#e5edf8"
    palette.highlight: "#2563eb"

    FolderDialog {
        id: workspaceDialog
        title: "Selecionar ou criar workspace TinyLogicVision"
        onAccepted: bridge.selectWorkspace(selectedFolder.toString())
    }

    ColumnLayout {
        anchors.fill: parent
        anchors.margins: 24
        spacing: 18

        RowLayout {
            Layout.fillWidth: true
            spacing: 14
            Rectangle {
                width: 52; height: 52; radius: 10; color: "#0f766e"
                Text { anchors.centerIn: parent; text: "TV"; color: "white"; font.bold: true; font.pixelSize: 20 }
            }
            ColumnLayout {
                spacing: 2
                Label { text: "TinyLogicVision"; font.pixelSize: 24; font.bold: true }
                Label { text: "Workspace Desktop · Sentinel-2 B2/B3/B4/B8"; color: "#8da2bd" }
            }
            Item { Layout.fillWidth: true }
            Rectangle {
                width: 12; height: 12; radius: 6
                color: bridge.serverRunning ? "#10b981" : "#64748b"
            }
            Label { text: bridge.serverRunning ? "Servidor ativo" : "Servidor parado"; color: "#a9b8cc" }
        }

        Frame {
            Layout.fillWidth: true
            padding: 18
            background: Rectangle { color: "#0d1829"; radius: 10; border.color: "#20314a" }
            ColumnLayout {
                anchors.fill: parent
                spacing: 10
                Label { text: "Workspace ativo"; font.bold: true; font.pixelSize: 15 }
                RowLayout {
                    Layout.fillWidth: true
                    TextField {
                        id: workspaceField
                        Layout.fillWidth: true
                        text: bridge.workspace
                        readOnly: bridge.serverRunning
                        onEditingFinished: bridge.selectWorkspace(text)
                    }
                    Button { text: "Selecionar pasta…"; enabled: !bridge.serverRunning; onClicked: workspaceDialog.open() }
                    Button { text: "Validar / criar"; enabled: !bridge.serverRunning; onClicked: bridge.initializeWorkspace() }
                }
                Label {
                    text: "Dados, modelos, execuções e proveniência permanecem isolados neste diretório."
                    color: "#8da2bd"
                    wrapMode: Text.WordWrap
                    Layout.fillWidth: true
                }
                RowLayout {
                    Layout.fillWidth: true
                    Label { text: "Recentes:"; color: "#8da2bd" }
                    ComboBox {
                        Layout.fillWidth: true
                        model: bridge.recentWorkspaces
                        enabled: !bridge.serverRunning && count > 0
                        onActivated: bridge.selectWorkspace(currentText)
                    }
                }
            }
        }

        Frame {
            Layout.fillWidth: true
            padding: 18
            background: Rectangle { color: "#0d1829"; radius: 10; border.color: "#20314a" }
            ColumnLayout {
                anchors.fill: parent
                spacing: 12
                Label { text: "Workbench científico"; font.bold: true; font.pixelSize: 15 }
                Label { text: bridge.serverUrl || "O endereço local será definido ao iniciar."; color: "#67e8f9" }
                RowLayout {
                    Button {
                        text: bridge.serverRunning ? "Servidor ativo" : "Iniciar servidor"
                        enabled: !bridge.serverRunning
                        highlighted: true
                        onClicked: bridge.startServer()
                    }
                    Button { text: "Abrir interface Web"; enabled: bridge.serverRunning; onClicked: bridge.openWorkbench() }
                    Button { text: "Encerrar"; enabled: bridge.serverRunning; onClicked: bridge.stopServer() }
                    Item { Layout.fillWidth: true }
                    Label { text: bridge.status; color: bridge.serverRunning ? "#34d399" : "#a9b8cc" }
                }
            }
        }

        Frame {
            Layout.fillWidth: true
            Layout.fillHeight: true
            padding: 12
            background: Rectangle { color: "#050a12"; radius: 8; border.color: "#20314a" }
            ScrollView {
                anchors.fill: parent
                TextArea {
                    text: bridge.log || "O log do servidor aparecerá aqui."
                    readOnly: true
                    color: "#9fb2c9"
                    font.family: "monospace"
                    font.pixelSize: 12
                    wrapMode: TextArea.WrapAnywhere
                    background: null
                }
            }
        }
    }
}
