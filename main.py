# -*- coding: utf-8 -*-
import base64
import os
import shutil
import random
import joblib
import jieba
import jieba.analyse
import numpy as np
from joblib import load
from sklearn.svm import SVC
from sklearn.feature_extraction import DictVectorizer

import dash
from dash import html, dcc
import feffery_antd_components as fac
from dash.dependencies import Input, Output, State
from dash_bootstrap_components import Modal, ModalHeader, ModalBody, ModalFooter

#
# # 数据集总文件夹路径
# path = 'D:\\数据集\\THUCNews\\THUCNews'
#
# # 定义要选择的文件夹列表
# class_list = {'财经', '房产', '社会',  '时尚', '教育', '时政', '游戏', '娱乐','彩票','股票','家居','科技','体育','星座'}
#
# # # 一
# # 为训练数据和测试数据创建总文件夹
train_data_path = '训练集'
test_data_path = '测试集'
# # os.makedirs(train_data_path, exist_ok=True)
# # os.makedirs(test_data_path, exist_ok=True)
#
# # 遍历每个类别的文件夹
# for folder_name in os.listdir(path):
#     if folder_name in class_list:
#         folder_path = os.path.join(path, folder_name)
#
        # # 获取所有txt文件
        # txt_files = [f for f in os.listdir(folder_path) if f.endswith('.txt')]
        #
        # if len(txt_files) >= 5000:
        #     # 随机选择5000个文件作为训练数据
        #     train_files = random.sample(txt_files, 5000)
        # else:
        #     train_files = random.sample(txt_files, len(txt_files)-1000)
        #
        # # 计算剩余的文件数量
        # remaining_files = [f for f in txt_files if f not in train_files]
        # # 检查剩余文件数量是否足够
        # if len(remaining_files) >= 3000:
        #     # 如果足够，则随机选择3000个不同的文件作为测试数据
        #     test_files = random.sample(remaining_files, 3000)
#         else:
#             # 如果不够，则减少要抽取的样本数量或处理其他逻辑
#             test_files = random.sample(remaining_files, len(remaining_files))
#
#         # 为这个文件夹在训练数据和测试数据文件夹中创建子文件夹
#         train_folder = os.path.join(train_data_path, folder_name)
#         test_folder = os.path.join(test_data_path, folder_name)
#         os.makedirs(train_folder, exist_ok=True)
#         os.makedirs(test_folder, exist_ok=True)
#
#         # 复制文件到训练数据文件夹
#         for train_file in train_files:
#             shutil.copy2(os.path.join(folder_path, train_file), train_folder)
#
#         # 复制文件到测试数据文件夹
#         for test_file in test_files:
#             shutil.copy2(os.path.join(folder_path, test_file), test_folder)
#
# print("文件复制完成")




# # 二 数据预处理
#
# 读取停用词列表
with open('stop_words_ch.txt', 'r', encoding='utf-8') as file:
    stop_words = set(word.strip() for word in file.readlines())
# #
# #
# # 数据预处理：分词、去除停用词
# def remove_stopwords_from_file(file_path, stop_words):
#     # 读取文件内容
#     with open(file_path, 'r', encoding='utf-8') as file:
#         content = file.read()
#     # jieba分词
#     words = jieba.cut(content)
#     # 去除停用词
#     words = [word for word in words if word not in stop_words]
#     # 将处理后的文本保存回文件
#     with open(file_path, 'w', encoding='utf-8') as file:
#         file.write(' '.join(words))
#
# # 遍历训练集每个子文件夹
# for folder_name in os.listdir(train_data_path):
#     folder_path = os.path.join(train_data_path, folder_name)
#     if os.path.isdir(folder_path):
#         # 遍历文件夹中的每个txt文件
#         for file_name in os.listdir(folder_path):
#             if file_name.endswith('.txt'):
#                 file_path = os.path.join(folder_path, file_name)
#                 # 去除文件中的停用词
#                 remove_stopwords_from_file(file_path, stop_words)
#
# # 遍历测试集每个子文件夹
# for folder_name in os.listdir(test_data_path):
#     folder_path = os.path.join(test_data_path, folder_name)
#     if os.path.isdir(folder_path):
#         # 遍历文件夹中的每个txt文件
#         for file_name in os.listdir(folder_path):
#             if file_name.endswith('.txt'):
#                 file_path = os.path.join(folder_path, file_name)
#                 # 去除文件中的停用词
#                 remove_stopwords_from_file(file_path, stop_words)
#
# print("数据处理完成")
#
#
# 三

# # 定义一个函数来提取每个类别的特征
# def extract_features(folder_path, topK=1000):
#     # 读取所有txt文件的内容
#     contents = ''
#     for file_name in os.listdir(folder_path):
#         if file_name.endswith('.txt'):
#             file_path = os.path.join(folder_path, file_name)
#             with open(file_path, 'r', encoding='utf-8') as file:
#                 contents += file.read()
#
#     # 使用jieba的TF/IDF方法提取关键词
#     keywords = jieba.analyse.extract_tags(contents, topK=topK, withWeight=True)
#
#     # 按照权重排序
#     keywords.sort(key=lambda x: x[1], reverse=True)
#
#     # 提取关键词和权重
#     words, weights = zip(*keywords)
#     return words, weights
#
# # 遍历训练集每个子文件夹
# for folder_name in os.listdir(train_data_path):
#     folder_path = os.path.join(train_data_path, folder_name)
#     if os.path.isdir(folder_path):
#         # 提取特征
#         words, weights = extract_features(folder_path)
#         # # 输出结果
#         # print(f"类别 {folder_name} 的前1000个关键词（按权重排序）:")
#         # for word, weight in zip(words, weights):
#         #     print(f"{word}: {weight}")
#
#         # 保存结果到文件
#         with open(os.path.join(folder_path, 'keywords.txt'), 'w', encoding='utf-8') as file:
#             for word, weight in zip(words, weights):
#                 file.write(f"{word}: {weight}\n")
# print("特征提取完成")



# 读取关键词文件并转换为特征向量
def read_keywords(directory):
    features = []
    labels = []
    for category in os.listdir(directory):
        category_folder = os.path.join(directory, category)
        if os.path.isdir(category_folder):
            keywords_file = os.path.join(category_folder, 'keywords.txt')
            if os.path.isfile(keywords_file):
                with open(keywords_file, 'r', encoding='utf-8') as file:
                    keywords = file.readlines()
                    # 关键词和权重是以冒号分隔的
                    keyword_dict = {word.split(':')[0]: float(word.split(':')[1].strip()) for word in keywords}
                    features.append(keyword_dict)
                    labels.append(category)
    return features, labels

train_features, train_labels = read_keywords('训练集')


# 表示第一个关键词及权重
# keyword,value =list(train_features[0].items())[0]
# print(keyword,value)


# # 创建DictVectorizer实例来将字典格式的特征转换为SVM模型所需的格式
# vectorizer = DictVectorizer()
# X_train = vectorizer.fit_transform(train_features)
#
# # 创建SVM模型
# model = SVC(kernel='linear')
#
# # 训练模型
# model.fit(X_train, train_labels)
#
# # 保存模型
# joblib.dump(model, 'svm_model.pkl')
#
# # 保存DictVectorizer
# joblib.dump(vectorizer, 'dict_vectorizer.pkl')


# 测试
# 加载已经训练好的模型
model = load('svm_model.pkl')
#
# 加载DictVectorizer
vectorizer = joblib.load('dict_vectorizer.pkl')
#
# # 遍历测试集的每个子文件夹和文件
# true_labels = []  # 用于存储真实标签
# predicted_labels = []  # 用于存储预测标签
# # i=0
# for category in os.listdir(test_data_path):
#     category_folder = os.path.join(test_data_path, category)
#     if os.path.isdir(category_folder):
#         for file_name in os.listdir(category_folder):
#             file_path = os.path.join(category_folder, file_name)
#             # 读取文件内容
#             with open(file_path, 'r', encoding='utf-8') as file:
#                 content = file.read()
#             # 特征提取
#             features = vectorizer.transform({file_name: content})
#             # 使用模型进行预测
#             prediction = model.predict(features)
#             predicted_labels.append(prediction[0])
#             # 存储真实标签
#             true_labels.append(category)
#
# # 计算准确度
# accuracy = sum(1 for true, predicted in zip(true_labels, predicted_labels) if true == predicted) / len(true_labels)
# print(f'模型准确度为：{accuracy:.2f}')
#


# 有内容时对新文章进行的处理
def process_article_content(content, stop_words):
    # 使用jieba进行分词
    words = jieba.cut(content)

    # 去除停用词
    words_filtered = [word for word in words if word not in stop_words and word.strip() != '']

    # 构建关键词字典
    keywords_dict = {}
    for word in words_filtered:
        if word in keywords_dict:
            keywords_dict[word] += 1
        else:
            keywords_dict[word] = 1
    # 特征提取
    features = vectorizer.transform(keywords_dict)
    # 使用模型进行预测
    prediction = model.predict(features)
    # print(f'新闻文章分类为: {prediction}')

    return prediction


# 有地址时对新文章进行的处理
def process_article(article_path, stop_words):
    # 读取文章内容
    with open(article_path, 'r', encoding='utf-8') as file:
        content = file.read()

    prediction=process_article_content(content, stop_words)
    return prediction

# # 示例
# article_path = '随机数据集\\215703.txt'
# process_article(article_path, stop_words)
#
# # 示例
# # 文章内容
# article_content = "翠林漫步百米密林 独享大自然恩宠 总有一些事物，拥有得天独厚的优势，获得不一样的恩宠。翠林漫步 (论坛 相册 户型 样板间 点评 地图搜索) (论坛 相册 户型 样板间 点评 地图搜索)紧守CSD中央休闲购物区核心，唱响未来，更以项目私属百米密林，展现它所拥有的不动声色的自然恩宠。私属密林，漫步时光上万颗郁郁葱葱的成树排列成百米原生森林，伫立起一道绿色生态走廊，将生活环抱于内。繁茂枝叶过滤了外围的喧嚣，只将大自然的声音留给居住在翠林漫步的人们，鸟儿清灵的鸣叫，知了“控告”夏日的“知了声”，花朵绽放的声音，绿草树木生长的声音，聚合成一首大自然生命之歌，萦绕在美好的生活之中。密麻婆娑、苍翠欲滴的成树静静的汲取水分、二氧化碳，转换成绿色鲜氧，供给您的纯氧呼吸，阳光穿透清晨的薄雾洒在这片土地之上，春花、夏露、秋实、冬雪的美景次第呈现，翠林漫步乐趣无限。社区内，翠林漫步以自然风光著称的英伦风景园为借鉴，秉承英国皇家造园的精诣技法，营建社区万米私家园林。大匠手笔的规划气度，以疏密有致的叠水、花台串联而成双景观主轴，四周分布鲜花草场、景观步道、亲水平台等公共游赏空间，追求人与景观的相容而一。草坪、树林和花卉自然排布，潺潺流水与原木廊桥向映，娇艳的玫瑰与柔软的草甸共生，一桥、一木、一亭、一台，处处流露着自然唯美的英式田园格调 (论坛 相册 户型 样板间 点评 地图搜索) (论坛 相册 户型 样板间 点评 地图搜索)，同时不失华丽与高雅，还有九篇园林小品点缀其中，凝练生活趣致，勾勒时光悠悠的步调，乐享慢生活。繁华核心，舒适生活 密林之外，是房山新城繁华之所在。CSD中央休闲购物区特有华北奥特莱斯旗舰店，超体量名品折扣中心，购物的天堂，还将发展成为集信息、文化、体育、会展、酒店、餐饮、休闲及办公为一体的城际中心，辐射全华北。更有城铁双高速，十余条公交构建通畅顺达的多维交通网；爱贝双语幼儿园、长阳中心小学、良乡小学、北师大附中、全国重点北京四中、北京理工大学、工商大学，孩子能享受一站式优质教育资源，家长后顾无忧；多种体育运动场，健身娱乐放松身心。翠林漫步为通透薄板，一梯两户，南北通透，全明格局，罕有全明厨明卫设计让居者真正乐享阳光生活，81-157平米丰富户型，动静明确，功能分明，空间收放自如，起居舒适有序。"
# process_article_content(article_content, stop_words)

#可视化页面

# 实例化Dash应用对象
app = dash.Dash(__name__)


# 创建布局
app.layout = html.Div([
    dcc.Upload(
        id='upload-data',
        children=html.Div([
            'Drag and Drop or ',
            html.A('Select Files')
        ]),
        style={
            'width': '100%',
            'height': '60px',
            'lineHeight': '60px',
            'borderWidth': '1px',
            'borderStyle': 'dashed',
            'borderRadius': '5px',
            'textAlign': 'center'
        },
        # Allow multiple files to be uploaded
        multiple=True
    ),
    html.Div(id='output-data-upload'),
    html.Div(id='submit-container'),
    html.Button(id='submit-button', n_clicks=0, children='Submit'),
    Modal(
        id='result-modal',
        children=[
            ModalHeader('结果'),
            ModalBody(html.Div(id='result-container')),
            ModalFooter(html.Button('关闭', id='close-modal-button'))
        ],
        is_open=False  # 默认不显示Modal
    )
])

# 处理文件上传
@app.callback(
    Output('output-data-upload', 'children'),
    Input('upload-data', 'contents'),
    State('upload-data', 'filename'),
    State('upload-data', 'last_modified')
)
def update_output(list_of_contents, list_of_names, list_of_dates):
    if list_of_contents is not None:
        children = [
            html.Div(
                [
                    html.Span(filename, style={'verticalAlign': 'middle'}),
                    html.I(className='fas fa-file-upload fa-lg'),
                ]
            )
            for filename in list_of_names
        ]
    else:
        children = []
    return children

# 处理提交按钮点击事件
@app.callback(
    [Output('result-container', 'children'),
     Output('result-modal', 'is_open')],
    [Input('submit-button', 'n_clicks'),
     Input('close-modal-button', 'n_clicks')],
    [State('upload-data', 'contents'),
     State('upload-data', 'filename')]
)
def display_output(submit_n_clicks, close_n_clicks, uploaded_content, uploaded_filename):
    # 判断是提交按钮还是关闭按钮触发了回调
    triggered_id = dash.callback_context.triggered[0]['prop_id'].split('.')[0]
    if triggered_id == 'submit-button' and submit_n_clicks > 0:
        if uploaded_content:
            # 上传文件的情况
            for content, filename in zip(uploaded_content, uploaded_filename):
                # 创建文件的绝对路径
                file_path = "随机数据集\\" + filename
                # print("文件路径：" + file_path)
                # 调用处理函数并返回结果
                result = process_article(file_path, stop_words)  # 假设process_article函数已经定义
                final_result=np.array_str(result)
                return html.Div('该文章的分类为: ' + final_result), True
        else:
            return html.Div("没有文件上传"), False
    elif triggered_id == 'close-modal-button' and close_n_clicks > 0:
        # 关闭弹窗
        return dash.no_update, False
    else:
        # 防止初始加载时Modal打开
        return dash.no_update, False

if __name__ == '__main__':
    app.run_server(debug=True)
