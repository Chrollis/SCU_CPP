#include "graphicscore.h"
#include "language_manager.h"
#include "le_net5.hpp"
#include "ui_graphicscore.h"
#include "vgg16.hpp"
#include <QFileDialog>
#include <QInputDialog>
#include <QMessageBox>
#include <QMouseEvent>
#include <QPaintEvent>
#include <QPainter>
#include <QSlider>

GraphicsCore::GraphicsCore(QWidget* parent)
    : QMainWindow(parent)
    , ui_(new Ui::GraphicsCore)
{
    ui_->setupUi(this);
    // 连接信号槽
    connect(this, &GraphicsCore::output_message, this, &GraphicsCore::update_output);
    connect(this, &GraphicsCore::train_button_text_changed, this, &GraphicsCore::update_train_button);
    connect(this, &GraphicsCore::status_message_changed, this, &GraphicsCore::update_status_message);
    connect(this, &GraphicsCore::training_finished, this, &GraphicsCore::handle_training_finished);
    connect(&language_manager::instance(), &language_manager::language_changed, this, &GraphicsCore::update_ui_language);
    // 连接语言功能
    connect(ui_->action_en_US, &QAction::triggered, this, [&]() { language_manager::instance().load_language("en_US"); });
    connect(ui_->action_zh_CN, &QAction::triggered, this, [&]() { language_manager::instance().load_language("zh_CN"); });
    connect(ui_->action_en_UK, &QAction::triggered, this, [&]() { language_manager::instance().load_language("en_UK"); });
    connect(ui_->action_fr_FR, &QAction::triggered, this, [&]() { language_manager::instance().load_language("fr_FR"); });
    connect(ui_->action_zh_TW, &QAction::triggered, this, [&]() { language_manager::instance().load_language("zh_TW"); });
    connect(ui_->action_ja_JP, &QAction::triggered, this, [&]() { language_manager::instance().load_language("ja_JP"); });
    connect(ui_->action_de_DE, &QAction::triggered, this, [&]() { language_manager::instance().load_language("de_DE"); });
    connect(ui_->action_ru_RU, &QAction::triggered, this, [&]() { language_manager::instance().load_language("ru_RU"); });
    connect(ui_->action_ko_KR, &QAction::triggered, this, [&]() { language_manager::instance().load_language("ko_KR"); });
    connect(ui_->action_es_ES, &QAction::triggered, this, [&]() { language_manager::instance().load_language("es_ES"); });
    connect(ui_->action_pt_BR, &QAction::triggered, this, [&]() { language_manager::instance().load_language("pt_BR"); });
    // 连接菜单和按钮动作
    connect(ui_->action_recognize, &QAction::triggered, this, &GraphicsCore::recognize_digits);
    connect(ui_->recognize, &QPushButton::clicked, this, &GraphicsCore::recognize_digits);
    connect(ui_->action_export, &QAction::triggered, this, &GraphicsCore::export_digits);
    connect(ui_->action_import_picture, &QAction::triggered, this, &GraphicsCore::import_picture);
    connect(ui_->picture_browse, &QPushButton::clicked, this, &GraphicsCore::import_picture);
    connect(ui_->action_about, &QAction::triggered, this, &GraphicsCore::show_about);
    connect(ui_->action_help, &QAction::triggered, this, &GraphicsCore::show_help);
    connect(ui_->action_close, &QAction::triggered, this, [&]() { save_settings(); exit(0); });
    connect(ui_->red, &QSlider::valueChanged, this, &GraphicsCore::update_color);
    connect(ui_->green, &QSlider::valueChanged, this, &GraphicsCore::update_color);
    connect(ui_->blue, &QSlider::valueChanged, this, &GraphicsCore::update_color);
    connect(ui_->action_save, &QAction::triggered, this, &GraphicsCore::save_settings);
    connect(ui_->action_clear_output, &QAction::triggered, ui_->output, &QTextEdit::clear);
    // 画布操作连接
    connect(ui_->action_clean, &QAction::triggered, this, &GraphicsCore::canvas_clean);
    connect(ui_->clean, &QPushButton::clicked, this, &GraphicsCore::canvas_clean);
    connect(ui_->action_redo, &QAction::triggered, this, &GraphicsCore::canvas_redo);
    connect(ui_->redo, &QPushButton::clicked, this, &GraphicsCore::canvas_redo);
    connect(ui_->action_undo, &QAction::triggered, this, &GraphicsCore::canvas_undo);
    connect(ui_->undo, &QPushButton::clicked, this, &GraphicsCore::canvas_undo);
    connect(ui_->action_clear, &QAction::triggered, this, &GraphicsCore::canvas_clear);
    // 模型操作连接
    connect(ui_->action_new_model, &QAction::triggered, this, &GraphicsCore::create_model);
    connect(ui_->action_open_model, &QAction::triggered, this, &GraphicsCore::load_model);
    connect(ui_->model_browse, &QPushButton::clicked, this, &GraphicsCore::load_model);
    connect(ui_->action_save_as, &QAction::triggered, this, &GraphicsCore::save_model_as);
    connect(ui_->model_train, &QPushButton::clicked, this, &GraphicsCore::model_train);
    connect(ui_->action_train_detailed, &QAction::triggered, this, [&]() { train_model(1); });
    connect(ui_->action_train_simple, &QAction::triggered, this, [&]() { train_model(0); });
    connect(ui_->action_stop_train, &QAction::triggered, this, &GraphicsCore::stop_train);
    // 数据文件浏览连接
    connect(ui_->train_data_browse, &QPushButton::clicked, this, &GraphicsCore::train_data_browse);
    connect(ui_->train_label_browse, &QPushButton::clicked, this, &GraphicsCore::train_label_browse);
    connect(ui_->test_data_browse, &QPushButton::clicked, this, &GraphicsCore::test_data_browse);
    connect(ui_->test_label_browse, &QPushButton::clicked, this, &::GraphicsCore::test_label_browse);
    // 初始化
    this->startTimer(33); // 定时器，33ms刷新一次
    settings_ = new QSettings("./config.ini", QSettings::IniFormat);
    load_settings(); // 加载配置文件
    this->setWindowTitle(chr::tr("title.application"));
    status_message_ = new QLabel(this);
    ui_->statusbar->addWidget(status_message_);
    // 画布事件过滤和属性设置
    ui_->canvas->installEventFilter(this);
    ui_->canvas->setAttribute(Qt::WA_StaticContents);
    ui_->canvas->setMouseTracking(1);
    // 初始化画布
    canvas_ = QImage(ui_->canvas->size(), QImage::Format_RGB32);
    canvas_.fill(Qt::white);
    update_color();
}

GraphicsCore::~GraphicsCore()
{
    delete settings_;
    delete ui_;
}
// 更新画笔颜色
void GraphicsCore::update_color()
{
    int r = ui_->red->value();
    int g = ui_->green->value();
    int b = ui_->blue->value();
    color_ = QColor(r, g, b);
    ui_->label_brush_color->setText(QString("RGB(%1,%2,%3)").arg(r).arg(g).arg(b));
    QPixmap color_block(20, 20);
    color_block.fill(color_);
    ui_->brush_color->setPixmap(color_block);
    need_update_ = 1;
}
// 画布撤销操作
void GraphicsCore::canvas_undo()
{
    if (undo_buffer_.size() > 0) {
        redo_buffer_.push_front(std::move(canvas_));
        canvas_ = std::move(undo_buffer_.front());
        undo_buffer_.pop_front();
        if (redo_buffer_.size() > 16) {
            redo_buffer_.pop_back();
        }
        need_update_ = 1;
        output_log(chr::tr("canvas.operation.undone"));
    }
}
// 画布重做操作
void GraphicsCore::canvas_redo()
{
    if (redo_buffer_.size() > 0) {
        undo_buffer_.push_front(std::move(canvas_));
        canvas_ = std::move(redo_buffer_.front());
        redo_buffer_.pop_front();
        if (undo_buffer_.size() > 16) {
            undo_buffer_.pop_back();
        }
        need_update_ = 1;
        output_log(chr::tr("canvas.operation.redone"));
    }
}
// 清空画布
void GraphicsCore::canvas_clean()
{
    undo_buffer_.push_front(canvas_.copy());
    canvas_.fill(Qt::white);
    redo_buffer_.clear();
    if (undo_buffer_.size() > 16) {
        undo_buffer_.pop_back();
    }
    need_update_ = 1;
    output_log(chr::tr("canvas.operation.cleared"));
}
// 清空画布历史记录
void GraphicsCore::canvas_clear()
{
    undo_buffer_.clear();
    redo_buffer_.clear();
    output_log(chr::tr("canvas.operation.history_cleared"));
}
// 识别数字
void GraphicsCore::recognize_digits()
{
    if (training_.isRunning()) {
        QMessageBox::warning(this, chr::tr("title.warning"), chr::tr("model.train.in_progress"));
        return;
    }
    cv::Mat src = chr::image_process::qimage_to_cv_mat(canvas_);
    digits_ = chr::image_process::process_image(src); // 图像处理提取数字区域
    if (digits_.empty()) {
        output_log(chr::tr("recognize.result").arg(chr::tr("recognize.nothing_found")));
        return;
    } else {
        digit_labels_.clear();
        src = chr::image_process::labelize_image(src, digits_);
        QImage dst = chr::image_process::cv_mat_to_qimage(src);
        QPainter painter(&dst);
        painter.setBrush(Qt::yellow);
        std::ostringstream oss;
        for (const auto& digit : digits_) {
            size_t label = model_->predict(model_->forward({ digit.data })); // 使用模型预测数字
            digit_labels_.push_back(label);
            oss << label;
            painter.setPen(Qt::NoPen); // 在画布上绘制识别结果
            painter.drawRect(digit.rect.x, digit.rect.y + digit.rect.height - 15, 8, 15);
            painter.setPen(Qt::SolidLine);
            painter.drawText(digit.rect.x, digit.rect.y + digit.rect.height, QString("%1").arg(label));
        }
        undo_buffer_.push_front(canvas_.copy());
        canvas_ = std::move(dst);
        redo_buffer_.clear();
        if (undo_buffer_.size() > 16) {
            undo_buffer_.pop_back();
        }
        need_update_ = 1;
        output_log(chr::tr("recognize.result").arg(oss.str()));
    }
}
// 导出识别到的数字
void GraphicsCore::export_digits()
{
    if (digits_.empty()) {
        QMessageBox::information(this, chr::tr("title.information"), chr::tr("recognize.no_digits"));
        return;
    }
    QString path = QFileDialog::getExistingDirectory(this, chr::tr("dialog.export.directory"), ".");
    if (path.isEmpty())
        return;
    for (size_t i = 0; i < digits_.size(); i++) {
        cv::Mat mat = chr::image_process::eigen_matrix_to_cv_mat(digits_[i].data);
        cv::Mat resized;
        cv::resize(mat, resized, cv::Size(160, 160));
        cv::imshow(std::to_string(digit_labels_[i]), resized);
        bool flag = 0; // 询问用户正确的标签
        size_t real = static_cast<size_t>(QInputDialog::getInt(this, chr::tr("dialog.export.correct_label"), chr::tr("dialog.export.correct_label_prompt"), static_cast<int>(digit_labels_[i]), 0, 9, 1, &flag));
        if (!flag) {
            cv::imwrite(QString("%1/%2-%3.png").arg(path).arg(digit_labels_[i]).arg(digits_[i].hash).toStdString(), mat);
        } else {
            cv::imwrite(QString("%1/%2-%3.png").arg(path).arg(real).arg(digits_[i].hash).toStdString(), mat);
        }
        cv::destroyWindow(std::to_string(digit_labels_[i]));
    }
    output_log(chr::tr("export.success").arg(path));
}
// 导入图片
void GraphicsCore::import_picture()
{
    QString path = QFileDialog::getOpenFileName(this, chr::tr("dialog.import.picture"), ".", chr::tr("file.filter.images"));
    if (path.isEmpty())
        return;
    ui_->picture_path->setText(path);
    QImage image(path);
    if (image.isNull()) {
        QMessageBox::warning(this, chr::tr("title.error"), chr::tr("import.picture.failed"));
        return;
    }
    undo_buffer_.push_front(canvas_.copy());
    image = image.scaled(ui_->canvas->size(), Qt::KeepAspectRatio, Qt::SmoothTransformation);
    QPainter painter(&canvas_);
    painter.drawImage(0, 0, image);
    redo_buffer_.clear();
    if (undo_buffer_.size() > 16) {
        undo_buffer_.pop_back();
    }
    need_update_ = 1;
    output_log(chr::tr("import.picture.success").arg(path));
}
// 创建模型
void GraphicsCore::create_model()
{
    if (training_.isRunning()) {
        QMessageBox::warning(this, chr::tr("title.warning"), chr::tr("model.train.in_progress"));
        return;
    }
    QString path = QFileDialog::getSaveFileName(this, chr::tr("model.io.create"), ".", chr::tr("file.filter.cnn_models"));
    if (path.isEmpty()) {
        return;
    }
    // 确保文件名以.cnn结尾
    if (!path.endsWith(".cnn", Qt::CaseInsensitive)) {
        path += ".cnn";
    }
    ui_->model_path->setText(path);
    save_settings();
    try {
        QStringList items;
        items << "LeNet-5" << "VGG16";
        bool flag = 0;
        std::string model_type = QInputDialog::getItem(this, chr::tr("dialog.model.type"), chr::tr("dialog.please.choose.one"), items, 0, 0, &flag).toStdString();
        if (!flag) {
            return;
        }
        if (model_type == "LeNet-5") {
            model_ = std::make_unique<chr::le_net5>();
        } else if (model_type == "VGG16") {
            model_ = std::make_unique<chr::vgg16>();
        } else {
            throw std::runtime_error(chr::tr("model.errors.invalid_type").arg(model_type).toStdString());
        }
        // 保存新模型
        connect(model_.get(), &chr::cnn_base::inform, this, &GraphicsCore::model_inform);
        connect(model_.get(), &chr::cnn_base::train_details, this, &GraphicsCore::model_train_details);
        model_->save(path.toStdString());
        ui_->model_type->setText(model_type.c_str());

    } catch (const std::exception& e) {
        QMessageBox::warning(this, chr::tr("title.error"), chr::tr("model.errors.create_failed").arg(e.what()));
        model_.reset(); // 重置模型指针
        ui_->model_type->setText("");
    }
}
// 加载模型
void GraphicsCore::load_model()
{
    QString path = QFileDialog::getOpenFileName(this, chr::tr("model.io.open"), ".", chr::tr("file.filter.cnn_models"));
    if (path.isEmpty())
        return;
    ui_->model_path->setText(path);
    save_settings();
    try {
        std::string model_type = chr::cnn_base::model_type_of(path.toStdString());
        if (model_type == "LeNet-5") {
            model_ = std::make_unique<chr::le_net5>();
        } else if (model_type == "VGG16") {
            model_ = std::make_unique<chr::vgg16>();
        } else {
            throw std::runtime_error(chr::tr("model.errors.invalid_type").arg(model_type).toStdString());
        }
        if (model_) {
            connect(model_.get(), &chr::cnn_base::inform, this, &GraphicsCore::model_inform);
            connect(model_.get(), &chr::cnn_base::train_details, this, &GraphicsCore::model_train_details);
            model_->load(path.toStdString()); // 加载已有模型
            ui_->model_type->setText(model_type.c_str());
        }
    } catch (const std::exception& e) {
        QMessageBox::warning(this, chr::tr("title.error"), chr::tr("model.errors.load_failed").arg(e.what()));
        model_.reset();
        ui_->model_type->setText("");
    }
}
// 模型另存为
void GraphicsCore::save_model_as()
{
    if (!model_) {
        QMessageBox::warning(this, chr::tr("title.warning"), chr::tr("model.errors.no_model"));
        return;
    }
    QString path = QFileDialog::getSaveFileName(this, chr::tr("model.io.save_as"), ".", chr::tr("file.filter.cnn_models"));
    if (path.isEmpty()) {
        return;
    }
    // 确保文件名以.cnn结尾
    if (!path.endsWith(".cnn", Qt::CaseInsensitive)) {
        path += ".cnn";
    }
    ui_->model_path->setText(path);
    save_settings();
    try {
        model_->save(path.toStdString());
    } catch (const std::exception& e) {
        QMessageBox::warning(this, chr::tr("title.error"), chr::tr("model.errors.save_failed").arg(e.what()));
    }
}
// 训练模型
void GraphicsCore::train_model(bool show_detail)
{
    if (!model_) {
        QMessageBox::warning(this, chr::tr("title.warning"), chr::tr("model.errors.no_model_for_train"));
        return;
    }
    if (training_.isRunning()) {
        QMessageBox::warning(this, chr::tr("title.warning"), chr::tr("model.train.in_progress"));
        return;
    }
    try {
        bool flag = 0; // 训练模型
        double target = QInputDialog::getDouble(this, chr::tr("dialog.train.parameters"), chr::tr("dialog.train.target_accuracy"), 0, 0, 100, 4, &flag);
        if (!flag)
            return;
        ui_->model_train->setText(chr::tr("button.train.stop"));
        is_training_ = 1;
        // 在后台线程中执行训练
        training_ = QtConcurrent::run([this, target, show_detail]() {
            emit output_message(chr::tr("train.data.loading"));
            auto train_dataset = chr::mnist_data::obtain_data(
                ui_->train_data->text().toStdString(),
                ui_->train_label->text().toStdString());
            std::vector<std::vector<chr::mnist_data>> train_batches;
            std::vector<chr::mnist_data> batch;
            size_t batch_size = ui_->batch->text().toULongLong();
            // 分批处理训练数据
            for (auto& data : train_dataset) {
                batch.push_back(std::move(data));
                if (batch.size() >= batch_size) {
                    train_batches.push_back(std::move(batch));
                    batch.clear();
                }
            }
            if (batch.size() > 0) {
                train_batches.push_back(std::move(batch));
            }
            emit output_message(chr::tr("train.data.loaded"));
            emit output_message(chr::tr("test.data.loading"));
            auto test_dataset = chr::mnist_data::obtain_data(
                ui_->test_data->text().toStdString(),
                ui_->test_label->text().toStdString());
            emit output_message(chr::tr("test.data.loaded"));
            emit output_message(chr::tr("model.train.starting").arg(target));
            QMetaObject::invokeMethod(this, "save_settings", Qt::QueuedConnection);
            size_t epochs = ui_->epoch->text().toULongLong();
            double learning_rate = ui_->learning_rate->text().toDouble();
            double accuracy = 0;
            size_t k = 0;
            // 训练循环，直到达到目标精度或取消训练
            do {
                accuracy = model_->train(train_batches[k], epochs, learning_rate, show_detail);
                k = (k + 1) % train_batches.size();
                model_->save(ui_->model_path->text().toStdString());
            } while (accuracy < target && !cancel_training_);
            if (!cancel_training_) {
                size_t correct = 0;
                emit output_message(chr::tr("evaluation.starting"));
                // 在测试集上评估模型
                for (size_t i = 0; i < test_dataset.size() && !cancel_training_; i++) {
                    auto vec = model_->forward({ test_dataset[i].image() });
                    if (model_->predict(vec) == test_dataset[i].label()) {
                        correct++;
                    }
                    if (show_detail) {
                        double progress = static_cast<double>(i + 1) / test_dataset.size() * 100.0;
                        QMetaObject::invokeMethod(this, "model_train_details",
                            Qt::QueuedConnection,
                            Q_ARG(double, progress),
                            Q_ARG(double, std::numeric_limits<double>::quiet_NaN()),
                            Q_ARG(size_t, correct),
                            Q_ARG(size_t, i + 1));
                    }
                }
                if (!cancel_training_) {
                    double final_accuracy = static_cast<double>(correct) / test_dataset.size() * 100.0;
                    emit output_message(chr::tr("evaluation.final_accuracy")
                            .arg(final_accuracy, 0, 'f', 2)
                            .arg(correct)
                            .arg(test_dataset.size()));
                }
            }
            emit training_finished();
        });
    } catch (const std::exception& e) {
        QMessageBox::warning(this, chr::tr("title.error"), chr::tr("error.train.failed").arg(e.what()));
        is_training_ = false;
        cancel_training_ = false;
        ui_->model_train->setText(chr::tr("button.train.start"));
    }
}
// 停止训练
void GraphicsCore::stop_train()
{
    if (!training_.isRunning()) {
        QMessageBox::information(this, chr::tr("title.information"), chr::tr("model.train.no_training"));
    } else {
        if (QMessageBox::question(this, chr::tr("title.question"), chr::tr("model.train.stop.confirm")) == QMessageBox::Yes) {
            cancel_training_ = 1;
            output_log(chr::tr("model.train.stop.requested"));
        }
    }
}
// 模型信息输出
void GraphicsCore::model_inform(const QString& output)
{
    output_log(output);
}
// 训练详情更新
void GraphicsCore::model_train_details(double progress, double loss, size_t correct, size_t total)
{
    std::ostringstream oss;
    int pos = static_cast<int>(40 * progress / 100.0);
    for (int j = 0; j < 40; ++j) {
        if (j < pos)
            oss << "=";
        else
            oss << " ";
    }
    double accuracy = static_cast<double>(correct) / total * 100.0;
    status_message_->setText(chr::tr("model.train.progress.status")
            .arg("[" + oss.str() + "]")
            .arg(progress, 0, 'f', 2)
            .arg(loss, 0, 'f', 4)
            .arg(accuracy, 0, 'f', 2)
            .arg(correct)
            .arg(total));
}
// 训练按钮点击处理
void GraphicsCore::model_train()
{
    if (!is_training_) {
        train_model(1);
    } else {
        stop_train();
    }
}
// 训练数据浏览
void GraphicsCore::train_data_browse()
{
    ui_->train_data->setText(QFileDialog::getOpenFileName(this, chr::tr("dialog.train.data.browse"), ".", chr::tr("file.filter.mnist_images")));
    save_settings();
}
// 训练标签浏览
void GraphicsCore::train_label_browse()
{
    ui_->train_label->setText(QFileDialog::getOpenFileName(this, chr::tr("dialog.train.labels.browse"), ".", chr::tr("file.filter.mnist_labels")));
    save_settings();
}
// 测试数据浏览
void GraphicsCore::test_data_browse()
{
    ui_->test_data->setText(QFileDialog::getOpenFileName(this, chr::tr("dialog.test.data.browse"), ".", chr::tr("file.filter.mnist_images")));
    save_settings();
}
// 测试标签浏览
void GraphicsCore::test_label_browse()
{
    ui_->test_label->setText(QFileDialog::getOpenFileName(this, chr::tr("dialog.test.labels.browse"), ".", chr::tr("file.filter.mnist_labels")));
    save_settings();
}
// 更新输出信息
void GraphicsCore::update_output(const QString& message)
{
    output_log(message);
}
// 更新训练按钮文本
void GraphicsCore::update_train_button(const QString& text)
{
    ui_->model_train->setText(text);
}
// 更新状态栏消息
void GraphicsCore::update_status_message(const QString& message)
{
    status_message_->setText(message);
}
// 训练完成处理
void GraphicsCore::handle_training_finished()
{
    status_message_->setText("");
    cancel_training_ = false;
    is_training_ = false;
    ui_->model_train->setText(chr::tr("button.train.start"));
}
// 事件过滤器，处理画布鼠标事件
bool GraphicsCore::eventFilter(QObject* obj, QEvent* event)
{
    if (obj == ui_->canvas) {
        if (event->type() == QEvent::MouseButtonPress) {
            QMouseEvent* mouse = static_cast<QMouseEvent*>(event);
            if (mouse->button() == Qt::LeftButton) {
                point_ = mouse->pos();
                mouse_down_ = 1;
                undo_buffer_.push_front(canvas_.copy());
                redo_buffer_.clear();
                if (undo_buffer_.size() > 16) {
                    undo_buffer_.pop_back();
                }
            }
            return 1;
        } else if (event->type() == QEvent::MouseMove && mouse_down_) {
            QMouseEvent* mouse = static_cast<QMouseEvent*>(event);
            draw_line_to(mouse->pos());
            return 1;
        } else if (event->type() == QEvent::MouseButtonRelease) {
            QMouseEvent* mouse = static_cast<QMouseEvent*>(event);
            if (mouse->button() == Qt::LeftButton && mouse_down_) {
                draw_line_to(mouse->pos());
                mouse_down_ = 0;
            }
            return 1;
        } else if (event->type() == QEvent::Paint) {
            QPaintEvent* paint = static_cast<QPaintEvent*>(event);
            QPainter painter(ui_->canvas);
            painter.fillRect(ui_->canvas->rect(), Qt::white);
            painter.drawImage(0, 0, canvas_);
            return 1;
        }
    }
    return QMainWindow::eventFilter(obj, event);
}
// 定时器事件，更新画布
void GraphicsCore::timerEvent(QTimerEvent* event)
{
    if (need_update_) {
        update();
        need_update_ = 0;
    }
}
// 关闭事件处理
void GraphicsCore::closeEvent(QCloseEvent* event)
{
    save_settings();
    if (training_.isRunning()) {
        if (QMessageBox::question(this, chr::tr("title.question"), chr::tr("application.close.training_warning")) == QMessageBox::Yes) {
            exit(0);
        }
    } else {
        exit(0);
    }
}
// 绘制线条
void GraphicsCore::draw_line_to(const QPoint& end_point)
{
    QPainter painter(&canvas_);
    painter.setRenderHint(QPainter::Antialiasing);
    painter.setPen(QPen(color_, 10, Qt::SolidLine, Qt::RoundCap, Qt::RoundJoin));
    painter.drawLine(point_, end_point);
    for (int i = 1; i <= 2; ++i) {
        QPoint intermediate_point = point_ + (end_point - point_) * i / (2 + 1);
        painter.drawPoint(intermediate_point);
    }
    point_ = end_point;
    need_update_ = 1;
}
// 输出日志
void GraphicsCore::output_log(const QString& output)
{
    QString timestamp = QDateTime::currentDateTime().toString("hh:mm:ss");
    ui_->output->append("[" + timestamp + "]: " + output);
}

void GraphicsCore::update_ui_language(bool changed, const QString& path)
{
    if (!changed) {
        QMessageBox::warning(this, chr::tr("title.warning"), chr::tr("error.change_language").arg(path));
        return;
    }
    // 更新窗口标题
    this->setWindowTitle(chr::tr("title.application"));

    // 更新菜单文本
    ui_->menu_file->setTitle(chr::tr("ui.menu.files"));
    ui_->menu_edit->setTitle(chr::tr("ui.menu.edit"));
    ui_->menu_train->setTitle(chr::tr("ui.menu.train"));
    ui_->menu_help->setTitle(chr::tr("ui.menu.help"));

    // 更新文件菜单项
    ui_->action_new_model->setText(chr::tr("ui.menu.new_model"));
    ui_->action_open_model->setText(chr::tr("ui.menu.open_model"));
    ui_->action_save_as->setText(chr::tr("ui.menu.save_as"));
    ui_->action_import_picture->setText(chr::tr("ui.button.import_picture"));
    ui_->action_export->setText(chr::tr("ui.menu.export"));
    ui_->action_close->setText(chr::tr("ui.menu.close"));

    // 更新编辑菜单项
    ui_->action_undo->setText(chr::tr("ui.button.undo"));
    ui_->action_redo->setText(chr::tr("ui.button.redo"));
    ui_->action_clean->setText(chr::tr("ui.button.clean_canvas"));
    ui_->action_clear->setText(chr::tr("ui.menu.clear_buffer"));
    ui_->action_recognize->setText(chr::tr("ui.button.recognize"));
    ui_->action_clear_output->setText(chr::tr("ui.menu.clear_output"));

    // 更新训练菜单项
    ui_->action_train_simple->setText(chr::tr("ui.menu.train_simple"));
    ui_->action_train_detailed->setText(chr::tr("ui.menu.train_detailed"));
    ui_->action_stop_train->setText(chr::tr("button.train.stop"));
    ui_->action_save->setText(chr::tr("ui.menu.save_config"));

    // 更新帮助菜单项
    ui_->action_help->setText(chr::tr("ui.menu.help_content"));
    ui_->action_about->setText(chr::tr("ui.menu.about"));
    ui_->menu_language->setTitle(chr::tr("ui.menu.language"));
    ui_->action_en_US->setText(chr::tr("ui.menu.language_american_english"));
    ui_->action_zh_CN->setText(chr::tr("ui.menu.language_simplified_chinese"));
    ui_->action_en_UK->setText(chr::tr("ui.menu.language_british_english"));
    ui_->action_fr_FR->setText(chr::tr("ui.menu.language_french"));
    ui_->action_zh_TW->setText(chr::tr("ui.menu.language_traditional_chinese"));
    ui_->action_ja_JP->setText(chr::tr("ui.menu.language_japanese"));
    ui_->action_de_DE->setText(chr::tr("ui.menu.language_german"));
    ui_->action_ru_RU->setText(chr::tr("ui.menu.language_russian"));
    ui_->action_ko_KR->setText(chr::tr("ui.menu.language_korean"));
    ui_->action_es_ES->setText(chr::tr("ui.menu.language_spanish"));
    ui_->action_pt_BR->setText(chr::tr("ui.menu.language_portuguese"));

    // 更新组框标题
    ui_->groupBox->setTitle(chr::tr("ui.group.canvas"));
    ui_->groupBox_2->setTitle(chr::tr("ui.group.model"));
    ui_->groupBox_3->setTitle(chr::tr("ui.group.output"));
    ui_->groupBox_4->setTitle(chr::tr("ui.group.painting"));

    // 更新模型组标签
    ui_->label_train_data->setText(chr::tr("ui.label.train_data"));
    ui_->label_train_label->setText(chr::tr("ui.label.train_label"));
    ui_->label_test_data->setText(chr::tr("ui.label.test_data"));
    ui_->label_test_label->setText(chr::tr("ui.label.test_label"));
    ui_->lable_model_path->setText(chr::tr("ui.label.model_path"));
    ui_->lable_model_type->setText(chr::tr("ui.label.model_type"));
    ui_->label_batch->setText(chr::tr("ui.label.batch_size"));
    ui_->label_epoch->setText(chr::tr("ui.label.epoch_times"));
    ui_->label_learning_rate->setText(chr::tr("ui.label.learning_rate"));

    // 更新绘画组标签和按钮
    ui_->undo->setText(chr::tr("ui.button.undo"));
    ui_->redo->setText(chr::tr("ui.button.redo"));
    ui_->clean->setText(chr::tr("ui.button.clean_canvas"));
    ui_->recognize->setText(chr::tr("ui.button.recognize"));
    ui_->label_import_picture->setText(chr::tr("ui.button.import_picture"));

    // 更新按钮文本
    ui_->train_data_browse->setText(chr::tr("ui.button.browse"));
    ui_->train_label_browse->setText(chr::tr("ui.button.browse"));
    ui_->test_data_browse->setText(chr::tr("ui.button.browse"));
    ui_->test_label_browse->setText(chr::tr("ui.button.browse"));
    ui_->model_browse->setText(chr::tr("ui.button.load"));
    ui_->picture_browse->setText(chr::tr("ui.button.browse"));

    // 更新训练按钮文本
    if (is_training_) {
        ui_->model_train->setText(chr::tr("button.train.stop"));
    } else {
        ui_->model_train->setText(chr::tr("button.train.start"));
    }

    save_settings();
}
// 加载设置
void GraphicsCore::load_settings()
{
    ui_->train_data->setText(settings_->value("train_data").toString());
    ui_->train_label->setText(settings_->value("train_label").toString());
    ui_->test_data->setText(settings_->value("test_data").toString());
    ui_->test_label->setText(settings_->value("test_label").toString());
    ui_->model_path->setText(settings_->value("model_path").toString());
    try {
        std::string model_type = chr::cnn_base::model_type_of(ui_->model_path->text().toStdString());
        if (model_type == "LeNet-5") {
            model_ = std::make_unique<chr::le_net5>();
        } else if (model_type == "VGG16") {
            model_ = std::make_unique<chr::vgg16>();
        } else {
            throw std::runtime_error(chr::tr("model.errors.invalid_type").arg(model_type).toStdString());
        }
        if (model_) {
            connect(model_.get(), &chr::cnn_base::inform, this, &GraphicsCore::model_inform);
            connect(model_.get(), &chr::cnn_base::train_details, this, &GraphicsCore::model_train_details);
            model_->load(ui_->model_path->text().toStdString()); // 加载已有模型
            ui_->model_type->setText(model_type.c_str());
        }
    } catch (const std::exception& e) {
        QMessageBox::warning(this, chr::tr("title.error"), chr::tr("model.errors.load_failed").arg(e.what()));
        model_.reset();
        ui_->model_type->setText("");
    }
    ui_->batch->setText(settings_->value("batch").toString());
    ui_->epoch->setText(settings_->value("epoch").toString());
    ui_->learning_rate->setText(settings_->value("learning_rate").toString());
    language_manager::instance().load_language(settings_->value("language").toString());
}
// 保存设置
void GraphicsCore::save_settings()
{
    settings_->setValue("train_data", ui_->train_data->text());
    settings_->setValue("train_label", ui_->train_label->text());
    settings_->setValue("test_data", ui_->test_data->text());
    settings_->setValue("test_label", ui_->test_label->text());
    settings_->setValue("model_path", ui_->model_path->text());
    settings_->setValue("batch", ui_->batch->text());
    settings_->setValue("epoch", ui_->epoch->text());
    settings_->setValue("learning_rate", ui_->learning_rate->text());
    settings_->setValue("language", language_manager::instance().current_language());
}

void GraphicsCore::show_about()
{
    QMessageBox::about(this,
        chr::tr("dialog.about.title"),
        chr::tr("about.content"));
}

// Help对话框
void GraphicsCore::show_help()
{
    QMessageBox::information(this,
        chr::tr("dialog.help.title"),
        chr::tr("help.content"));
}
