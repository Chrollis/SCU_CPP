#ifndef LANGUAGE_MANAGER_H
#define LANGUAGE_MANAGER_H

#include <QFile>
#include <QJsonDocument>
#include <QJsonObject>
#include <QMap>
#include <QObject>
#include <QString>

class language_manager : public QObject {
    Q_OBJECT
public:
    static language_manager& instance();
    void load_language(const QString& language_code);
    QString translate(const QString& key) const;
    QString current_language() const;
signals:
    void language_changed(bool changed, const QString& path);

private:
    language_manager(QObject* parent = nullptr);
    QJsonObject translations_;
    QString current_language_;
};

namespace chr {
inline QString tr(const QString& key)
{
    return language_manager::instance().translate(key);
}
}

#endif // LANGUAGE_MANAGER_H
