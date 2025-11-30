#include "language_manager.h"
#include <QMessageBox>
#include <filesystem>

language_manager& language_manager::instance()
{
    static language_manager instance;
    return instance;
}

void language_manager::load_language(const QString& language_code)
{
    std::filesystem::create_directory("./i18n");
    QString filename = QString("./i18n/%1.json").arg(language_code);
    QFile file(filename);
    if (!file.open(QIODevice::ReadOnly)) {
        emit language_changed(0, filename);
        return;
    }
    QByteArray data = file.readAll();
    QJsonDocument doc = QJsonDocument::fromJson(data);
    if (doc.isNull()) {
        emit language_changed(0, filename);
        return;
    }
    translations_ = doc.object();
    current_language_ = language_code;
    emit language_changed(1, filename);
}

QString language_manager::translate(const QString& key) const
{
    if (translations_.contains(key)) {
        return translations_[key].toString();
    }
    return key;
}

QString language_manager::current_language() const
{
    return current_language_;
}

language_manager::language_manager(QObject* parent)
    : QObject { parent }
{
    this->load_language("en_US");
}
