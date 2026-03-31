# 导入 Gradio 库，用于构建交互式前端界面
import gradio as gr
import os
# 导入 requests 库，用于发送 HTTP 请求
import requests
# 导入 json 库，用于处理 JSON 数据
import json
# 导入 logging 库，用于记录日志
import logging
import socket
# 导入 re 库，用于正则表达式操作
import re
# 导入 uuid 库，用于生成唯一标识符
import uuid
# 导入 datetime 库，用于处理日期和时间
from datetime import datetime

# 设置日志的基本配置，指定日志级别为 INFO，并定义日志格式
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
# 创建一个名为当前模块的日志记录器
logger = logging.getLogger(__name__)

# 定义后端服务接口的 URL 地址
url = "http://localhost:8012/v1/chat/completions"
# 定义 HTTP 请求头，指定内容类型为 JSON
headers = {"Content-Type": "application/json"}

# 是否流式输出
stream_flag = True  # False

# 初始化一个空字典，用于模拟用户数据库
users_db = {}
# 初始化一个空字典，用于存储用户名与用户 ID 的映射
user_id_map = {}


# 定义生成唯一用户 ID 的函数
def generate_unique_user_id(username):
    # 如果用户名不在映射表中，则生成一个新的 UUID
    if username not in user_id_map:
        user_id = str(uuid.uuid4())
        # 确保生成的 ID 未被使用，若重复则重新生成
        while user_id in user_id_map.values():
            user_id = str(uuid.uuid4())
        # 将用户名和生成的 ID 存入映射表
        user_id_map[username] = user_id
    # 返回该用户对应的唯一 ID
    return user_id_map[username]


# 定义生成唯一会话 ID 的函数
def generate_unique_conversation_id(username):
    # 返回由用户名和 UUID 拼接而成的会话 ID
    return f"{username}_{uuid.uuid4()}"


# 定义发送消息的函数，处理用户输入并获取后端回复
def send_message(user_message, history, user_id, conversation_id, username):
    # 构造发送给后端的数据，包含用户消息、用户 ID 和会话 ID
    data = {
        "messages": [{"role": "user", "content": user_message}],
        "stream": stream_flag,
        "userId": user_id,
        "conversationId": conversation_id
    }

    # 更新聊天历史，添加用户消息和临时占位回复
    history = history + [
        {"role": "user", "content": user_message},
        {"role": "assistant", "content": "正在生成回复..."}
    ]
    
    # 获取当前会话标题
    current_title = None
    if username and conversation_id and username in users_db and conversation_id in users_db[username]["conversations"]:
        current_title = users_db[username]["conversations"][conversation_id].get("title", "创建新的聊天")
    
    # 第一次 yield，返回当前的聊天历史和标题
    yield history, history, current_title

    # 如果是首次消息，设置会话标题为用户消息的前 20 个字符或完整消息
    if username and conversation_id:
        if username in users_db and conversation_id in users_db[username]["conversations"]:
            if not users_db[username]["conversations"][conversation_id].get("title_set", False):
                new_title = user_message[:20] if len(user_message) > 20 else user_message
                users_db[username]["conversations"][conversation_id]["title"] = new_title
                users_db[username]["conversations"][conversation_id]["title_set"] = True

    # 定义格式化回复内容的函数
    def format_response(full_text):
        # 将 <think> 标签替换为加粗的"思考过程"标题
        formatted_text = re.sub(r'<think>', '**思考过程**：\n', full_text)
        # 将 </think> 标签替换为加粗的"最终回复"标题
        formatted_text = re.sub(r'</think>', '\n\n**最终回复**：\n', formatted_text)
        # 返回去除前后空白的格式化文本
        return formatted_text.strip()

    # 流式输出
    if stream_flag:
        assistant_response = ""
        try:
            with requests.post(url, headers=headers, data=json.dumps(data), stream=True, timeout=60) as response:
                for line in response.iter_lines():
                    if line:
                        json_str = line.decode('utf-8').strip("data: ")
                        if not json_str:
                            logger.info("收到空字符串，跳过...")
                            continue
                        if json_str.startswith('{') and json_str.endswith('}'):
                            try:
                                response_data = json.loads(json_str)
                                if 'delta' in response_data['choices'][0]:
                                    content = response_data['choices'][0]['delta'].get('content', '')
                                    # 直接累积原始内容，不进行格式化
                                    logger.info(f"接收数据:{content}")
                                    assistant_response += content
                                    updated_history = history[:-1] + [
                                        {"role": "assistant", "content": assistant_response}
                                    ]
                                    yield updated_history, updated_history, current_title
                                if response_data.get('choices', [{}])[0].get('finish_reason') == "stop":
                                    logger.info("接收JSON数据结束")
                                    # 流结束时对完整响应进行最终格式化
                                    formatted_response = format_response(assistant_response)
                                    final_history = history[:-1] + [
                                        {"role": "assistant", "content": formatted_response}
                                    ]
                                    yield final_history, final_history, current_title
                                    break
                            except json.JSONDecodeError as e:
                                logger.error(f"JSON解析错误: {e}")
                                error_history = history[:-1] + [
                                    {"role": "assistant", "content": "解析响应时出错，请稍后再试。"}
                                ]
                                yield error_history, error_history, current_title
                                break
                        else:
                            logger.info(f"无效JSON格式: {json_str}")
                    else:
                        logger.info("收到空行")
                else:
                    logger.info("流式响应结束但未明确结束")
                    # 对不完整响应进行格式化
                    formatted_response = format_response(assistant_response) if assistant_response else "未收到完整响应。"
                    incomplete_history = history[:-1] + [
                        {"role": "assistant", "content": formatted_response}
                    ]
                    yield incomplete_history, incomplete_history, current_title
        except requests.RequestException as e:
            logger.error(f"请求失败: {e}")
            error_history = history[:-1] + [
                {"role": "assistant", "content": "请求失败，请稍后再试。"}
            ]
            yield error_history, error_history, current_title

    # 非流式输出
    else:
        # 获取当前会话标题
        current_title = None
        if username and conversation_id and username in users_db and conversation_id in users_db[username]["conversations"]:
            current_title = users_db[username]["conversations"][conversation_id].get("title", "创建新的聊天")
            
        try:
            # 向后端发送 POST 请求并获取响应
            response = requests.post(url, headers=headers, data=json.dumps(data), timeout=60)
            # 将响应解析为 JSON 格式
            response_json = response.json()
            # 提取助手的回复内容
            assistant_content = response_json['choices'][0]['message']['content']
            # 对助手回复进行格式化
            formatted_content = format_response(assistant_content)
            # 更新聊天历史，替换临时占位回复为格式化后的内容
            updated_history = history[:-1] + [
                {"role": "assistant", "content": formatted_content}
            ]
            # 第二次 yield，返回更新后的聊天历史和标题
            yield updated_history, updated_history, current_title
        except requests.RequestException as e:
            logger.error(f"请求失败: {e}")
            error_history = history[:-1] + [
                {"role": "assistant", "content": "请求失败，请稍后再试。"}
            ]
            yield error_history, error_history, current_title


# 定义注册用户的函数
def register(username, password):
    # 如果用户名已存在，返回错误提示
    if username in users_db:
        return "用户名已存在！"
    # 生成唯一用户 ID
    user_id = generate_unique_user_id(username)
    # 在用户数据库中添加新用户信息
    users_db[username] = {"password": password, "user_id": user_id, "conversations": {}}
    # 返回注册成功的提示
    return "注册成功！请关闭弹窗并登录。"


# 定义用户登录的函数
def login(username, password):
    # 检查用户名是否存在且密码匹配
    if username in users_db and users_db[username]["password"] == password:
        # 获取用户 ID
        user_id = users_db[username]["user_id"]
        # 生成新的会话 ID
        conversation_id = generate_unique_conversation_id(username)
        # 获取当前时间作为会话创建时间
        create_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        # 在用户数据库中添加新会话记录
        users_db[username]["conversations"][conversation_id] = {
            "history": [],
            "title": "创建新的聊天",
            "create_time": create_time,
            "title_set": False
        }
        # 返回登录成功的结果及相关信息，包括页面状态
        result = (
            True,  # logged_in
            username,  # current_user
            user_id,  # current_user_id
            conversation_id,  # current_conversation
            "登录成功！",  # login_output
            gr.update(visible=False),  # 隐藏登录页面
            gr.update(visible=False),  # 隐藏注册页面
            gr.update(visible=True),   # 显示聊天页面
            "chat"  # page_state
        )
        print(f"登录成功，返回: {result}")
        return result
    # 如果登录失败，返回错误提示和登录页面状态
    result = (
        False,  # logged_in
        None,  # current_user
        None,  # current_user_id
        None,  # current_conversation
        "用户名或密码错误！",  # login_output
        gr.update(visible=True),   # 显示登录页面
        gr.update(visible=False),  # 隐藏注册页面
        gr.update(visible=False),  # 隐藏聊天页面
        "login"  # page_state
    )
    print(f"登录失败，返回: {result}")
    return result


# 定义创建新会话的函数
def new_conversation(username):
    # 如果用户未登录，返回提示
    if username not in users_db:
        return "请先登录！", None
    # 生成新的会话 ID
    conversation_id = generate_unique_conversation_id(username)
    # 获取当前时间作为会话创建时间
    create_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    # 在用户数据库中添加新会话记录
    users_db[username]["conversations"][conversation_id] = {
        "history": [],
        "title": "创建新的聊天",
        "create_time": create_time,
        "title_set": False
    }
    # 返回成功提示和新会话 ID
    return "新会话创建成功！", conversation_id


# 定义获取会话列表的函数
def get_conversation_list(username):
    # 如果用户未登录或无会话记录，返回默认选项
    if username not in users_db or not users_db[username]["conversations"]:
        return ["请选择历史会话"]
    # 初始化会话列表
    conv_list = []
    # 遍历用户的所有会话
    for conv_id, details in users_db[username]["conversations"].items():
        # 获取会话标题，默认为"未命名会话"
        title = details.get("title", "未命名会话")
        # 获取会话创建时间
        create_time = details.get("create_time", datetime.now().strftime("%Y-%m-%d %H:%M:%S"))
        # 将标题和时间拼接后添加到列表
        conv_list.append(f"{title} - {create_time}")
    # 返回包含默认选项的会话列表
    return ["请选择历史会话"] + conv_list


# 定义从选项中提取会话 ID 的函数
def extract_conversation_id(selected_option, username):
    # 如果选择的是默认选项或用户未登录，返回 None
    if selected_option == "请选择历史会话" or username not in users_db:
        return None
    # 遍历用户的所有会话
    for conv_id, details in users_db[username]["conversations"].items():
        # 获取会话标题和创建时间
        title = details.get("title", "未命名会话")
        create_time = details.get("create_time", "")
        # 如果选项匹配，则返回对应的会话 ID
        if f"{title} - {create_time}" == selected_option:
            return conv_id
    # 如果未找到匹配项，返回 None
    return None


# 定义加载会话历史的函数
def load_conversation(username, selected_option):
    # 如果选择的是默认选项或用户未登录，返回空历史
    if selected_option == "请选择历史会话" or username not in users_db:
        return []
    # 从选项中提取会话 ID
    conversation_id = extract_conversation_id(selected_option, username)
    # 如果会话 ID 存在，返回对应的聊天历史
    if conversation_id and conversation_id in users_db[username]["conversations"]:
        return users_db[username]["conversations"][conversation_id]["history"]
    # 否则返回空历史
    return []


# 使用 Gradio Blocks 创建前端界面
with gr.Blocks(title="聊天助手", css="""
    .login-container, .register-container {
        max-width: 400px;
        margin: 0 auto;
        padding: 20px;
        border-radius: 10px;
        box-shadow: 0 4px 8px rgba(0,0,0,0.1);
    }
    .header {
        background-color: #f8f9fa;
        padding: 10px;
        border-radius: 5px;
        margin-bottom: 10px;
    }
    .header-btn {
        margin: 0 5px;
    }
    .chat-area {
        max-width: 800px;
        margin: 0 auto;
    }
    .modal {
        position: fixed;
        top: 50%;
        left: 50%;
        transform: translate(-50%, -50%);
        background-color: white;
        padding: 20px;
        border-radius: 10px;
        box-shadow: 0 4px 8px rgba(0,0,0,0.2);
        z-index: 1000;
        max-width: 400px;
        width: 90%;
    }
    .switch-prompt {
        text-align: center;
        margin: 10px 0;
    }
    .sending-indicator {
        display: inline-block;
        padding: 5px 10px;
        background-color: #e3f2fd;
        color: #1976d2;
        border-radius: 4px;
        font-size: 14px;
        margin-top: 5px;
    }
    #title-display {
        text-align: center;
        margin-bottom: 10px;
    }
""") as demo:
    # 定义状态变量，用于追踪登录状态
    logged_in = gr.State(False)
    # 定义状态变量，用于存储当前用户名
    current_user = gr.State(None)
    # 定义状态变量，用于存储当前用户 ID
    current_user_id = gr.State(None)
    # 定义状态变量，用于存储发送状态
    sending_state = gr.State(False)
    
    # 定义发送消息的函数，包含状态管理
    def send_message_with_state(user_message, chat_history, user_id, conversation_id, username, is_sending):
        # 如果正在发送中，忽略新的发送请求
        if is_sending:
            yield chat_history, chat_history, "创建新的聊天", True, gr.update(visible=True, value="<div class='sending-indicator'>正在发送中...</div>")
            return
        
        # 设置发送状态为True
        yield chat_history, chat_history, "创建新的聊天", True, gr.update(visible=True, value="<div class='sending-indicator'>正在发送中...</div>")
        
        # 调用原始的流式响应处理函数
        for response in handle_stream_response(user_message, chat_history, user_id, conversation_id, username):
            chatbot, chatbot_history, title = response
            yield chatbot, chatbot_history, title, True, gr.update(visible=True, value="<div class='sending-indicator'>正在接收回复...</div>")
        
        # 发送完成后重置状态
        yield chatbot, chatbot_history, title, False, gr.update(visible=False)
    current_conversation = gr.State(None)
    # 定义状态变量，用于存储聊天历史
    chatbot_history = gr.State([])
    # 定义状态变量，用于存储会话标题
    conversation_title = gr.State("创建新的聊天")

    # 定义页面状态变量，用于控制显示哪个页面
    page_state = gr.State("login")  # 默认显示登录页面
    
    # 定义登录页面布局，初始可见
    with gr.Column(visible=True, elem_classes="login-container") as login_page:
        # 显示标题
        gr.Markdown("## 聊天助手")
        # 定义用户名输入框
        login_username = gr.Textbox(label="用户名", placeholder="请输入用户名")
        # 定义密码输入框，隐藏输入内容
        login_password = gr.Textbox(label="密码", placeholder="请输入密码", type="password")
        # 创建一行布局放置登录和注册按钮
        with gr.Row():
            # 定义登录按钮
            login_button = gr.Button("登录", variant="primary")
            # 定义注册按钮
            register_button = gr.Button("注册", variant="secondary")
        # 定义登录结果输出框，不可编辑
        login_output = gr.Textbox(label="结果", interactive=False)
        
        # 添加页面切换提示
        gr.Markdown("还没有账号？", elem_classes="switch-prompt")
        switch_to_register = gr.Button("立即注册", variant="link", size="sm")

    # 定义注册页面布局，初始不可见
    with gr.Column(visible=False, elem_classes="register-container") as register_page:
        # 显示标题
        gr.Markdown("## 注册新账号")
        # 定义注册用户名输入框
        reg_username = gr.Textbox(label="用户名", placeholder="请输入用户名")
        # 定义注册密码输入框，隐藏输入内容
        reg_password = gr.Textbox(label="密码", placeholder="请输入密码", type="password")
        # 创建一行布局放置提交和返回按钮
        with gr.Row():
            # 定义提交注册按钮
            reg_button = gr.Button("提交注册", variant="primary")
            # 定义返回登录按钮
            switch_to_login = gr.Button("返回登录", variant="secondary", size="sm")
        # 定义注册结果输出框，不可编辑
        reg_output = gr.Textbox(label="结果", interactive=False)
        
        # 添加页面切换提示
        gr.Markdown("已有账号？", elem_classes="switch-prompt")
        switch_to_login_from_register = gr.Button("立即登录", variant="link", size="sm")

    # 定义聊天页面布局，初始不可见
    with gr.Column(visible=False) as chat_page:
        # 定义头部布局，包含欢迎文本和按钮
        with gr.Row(elem_classes="header"):
            # 显示欢迎文本，初始值为空
            welcome_text = gr.Markdown("### 欢迎，")
            # 创建一行布局放置头部按钮
            with gr.Row():
                # 定义新建会话按钮
                new_conv_button = gr.Button("新建会话", elem_classes="header-btn", variant="secondary")
                # 定义历史会话按钮
                history_button = gr.Button("历史会话", elem_classes="header-btn", variant="secondary")
                # 定义退出登录按钮
                logout_button = gr.Button("退出登录", elem_classes="header-btn", variant="secondary")

        # 定义聊天区域布局
        with gr.Column(elem_classes="chat-area"):
            # 显示会话标题
            title_display = gr.Markdown("## 会话标题", elem_id="title-display")
            # 定义聊天对话框，高度为 450 像素，指定消息格式
            chatbot = gr.Chatbot(label="聊天对话", height=450, type="messages")
            # 创建一行布局放置消息输入框和发送按钮
            with gr.Row():
                # 定义消息输入框
                message = gr.Textbox(label="消息", placeholder="输入消息并按 Enter 发送", scale=8, container=False)
                # 定义发送按钮
                send = gr.Button("发送", scale=2, variant="primary")
                # 定义发送状态指示
                sending_indicator = gr.HTML("", visible=False, elem_id="sending-indicator")

    # 定义历史会话弹窗布局，初始不可见
    with gr.Column(visible=False, elem_classes="modal") as history_modal:
        # 显示历史会话标题
        gr.Markdown("### 会话历史")
        # 定义会话下拉选择框，初始选项为"请选择历史会话"
        conv_dropdown = gr.Dropdown(label="选择历史会话", choices=["请选择历史会话"], value="请选择历史会话")
        # 定义加载会话按钮
        load_conv_button = gr.Button("加载会话", variant="primary")
        # 定义关闭历史弹窗按钮
        close_history_button = gr.Button("关闭", variant="secondary")

    # 定义页面切换函数
    def show_login_page():
        return (
            gr.update(visible=True),  # 显示登录页面
            gr.update(visible=False),  # 隐藏注册页面
            gr.update(visible=False),   # 隐藏聊天页面
            "login"  # 更新页面状态
        )
    
    def show_register_page():
        return (
            gr.update(visible=False),  # 隐藏登录页面
            gr.update(visible=True),   # 显示注册页面
            gr.update(visible=False),   # 隐藏聊天页面
            "register"  # 更新页面状态
        )
    
    def show_chat_page():
        print("执行show_chat_page函数")
        result = (
            gr.update(visible=False),  # 隐藏登录页面
            gr.update(visible=False),  # 隐藏注册页面
            gr.update(visible=True),   # 显示聊天页面
            "chat"  # 更新页面状态
        )
        print(f"show_chat_page返回: {result}")
        return result

    # 定义显示历史弹窗的函数，并更新会话列表
    def show_history_modal(username):
        return gr.update(visible=True), gr.update(choices=get_conversation_list(username), value="请选择历史会话")

    # 定义隐藏历史弹窗的函数
    def hide_history_modal():
        return gr.update(visible=False)

    # 定义退出登录的函数，重置所有状态
    def logout():
        return (
            False, None, None,
            gr.update(visible=True), gr.update(visible=False), gr.update(visible=False), "login",  # 显示登录页面
            "已退出登录", [], None, [], "创建新的聊天",
            gr.update(value=""), gr.update(value="")  # 清空用户名和密码输入框
        )

    # 定义更新欢迎文本的函数
    def update_welcome_text(username):
        return gr.update(value=f"### 欢迎，{username}")

    # 定义更新标题显示的函数
    def update_title_display(title):
        return gr.update(value=f"## {title}")

    # 绑定页面切换按钮事件
    register_button.click(show_register_page, None, [login_page, register_page, chat_page, page_state])
    switch_to_register.click(show_register_page, None, [login_page, register_page, chat_page, page_state])
    switch_to_login.click(show_login_page, None, [login_page, register_page, chat_page, page_state])
    switch_to_login_from_register.click(show_login_page, None, [login_page, register_page, chat_page, page_state])
    # 绑定提交注册按钮点击事件，调用注册函数
    reg_button.click(
        register, [reg_username, reg_password], reg_output
    ).then(
        # 注册成功后清空输入框
        lambda result: [gr.update(value=""), gr.update(value="")] if result and "注册成功" in result else [gr.update(value=""), gr.update(value="")],
        None,
        [reg_username, reg_password]
    ).then(
        # 注册成功后跳转到登录页面
        lambda result: show_login_page() if result and "注册成功" in result else (gr.update(visible=True), gr.update(visible=False), gr.update(visible=False), "login"),
        None,
        [login_page, register_page, chat_page, page_state]
    )

    # 绑定登录按钮点击事件，调用登录函数（包含页面切换）
    login_button.click(
        login, [login_username, login_password],
        [logged_in, current_user, current_user_id, current_conversation, login_output,
         login_page, register_page, chat_page, page_state]
    ).then(
        # 更新欢迎文本（仅在登录成功时）
        lambda logged_in, username: update_welcome_text(username) if logged_in else gr.update(value="### 欢迎，访客"),
        [logged_in, current_user], 
        welcome_text
    ).then(
        # 加载当前会话历史（仅在登录成功时）
        lambda logged_in, username, conv_id: users_db[username]["conversations"][conv_id]["history"]
        if logged_in and username and conv_id and username in users_db and conv_id in users_db.get(username, {}).get("conversations", {})
        else [],
        [logged_in, current_user, current_conversation], 
        chatbot_history
    ).then(
        # 更新会话标题（仅在登录成功时）
        lambda logged_in, username, conv_id: users_db[username]["conversations"][conv_id].get("title", "创建新的聊天")
        if logged_in and username and conv_id and username in users_db and conv_id in users_db.get(username, {}).get("conversations", {})
        else "创建新的聊天",
        [logged_in, current_user, current_conversation], 
        conversation_title
    ).then(
        # 更新标题显示
        update_title_display, 
        [conversation_title], 
        title_display
    )

    # 绑定退出登录按钮点击事件，调用退出函数
    logout_button.click(
        logout, None,
        [logged_in, current_user, current_user_id, login_page, register_page, chat_page, page_state,
         login_output, chatbot, current_conversation, chatbot_history, conversation_title,
         login_username, login_password]
    )

    # 绑定历史会话按钮点击事件，显示历史弹窗
    history_button.click(show_history_modal, [current_user], [history_modal, conv_dropdown])
    # 绑定关闭历史弹窗按钮点击事件，隐藏历史弹窗
    close_history_button.click(hide_history_modal, None, history_modal)

    # 绑定新建会话按钮点击事件，调用新建会话函数
    new_conv_button.click(
        new_conversation, [current_user], [current_conversation]
    ).then(
        # 清空聊天对话框
        lambda: [], None, chatbot
    ).then(
        # 清空聊天历史状态
        lambda: [], None, chatbot_history
    ).then(
        # 重置会话标题
        lambda: "创建新的聊天", None, conversation_title
    ).then(
        # 更新标题显示
        update_title_display, [conversation_title], title_display
    )

    # 绑定加载会话按钮点击事件，加载选中的会话历史
    load_conv_button.click(
        load_conversation, [current_user, conv_dropdown], [chatbot, chatbot_history]
    ).then(
        # 更新当前会话 ID
        lambda user, conv: extract_conversation_id(conv, user),
        [current_user, conv_dropdown], current_conversation
    ).then(
        # 更新会话标题
        lambda username, conv: users_db[username]["conversations"][
            extract_conversation_id(conv, username)
        ].get("title", "创建新的聊天")
        if username and conv and username in users_db and extract_conversation_id(conv, username)
        else "创建新的聊天",
        [current_user, conv_dropdown], conversation_title
    ).then(
        # 更新标题显示
        update_title_display, [conversation_title], title_display
    ).then(
        # 隐藏历史弹窗
        hide_history_modal, None, history_modal
    )

    # 定义更新聊天历史的函数
    def update_history(chatbot_output, history, user, conv_id):
        # 如果用户和会话 ID 存在，更新数据库中的聊天历史
        if user and conv_id and user in users_db and conv_id in users_db.get(user, {}).get("conversations", {}):
            users_db[user]["conversations"][conv_id]["history"] = chatbot_output
        return chatbot_output

    # 定义安全获取会话标题的函数
    def safe_get_title(username, conv_id):
        if (username and conv_id and username in users_db
                and conv_id in users_db.get(username, {}).get("conversations", {})):
            return users_db[username]["conversations"][conv_id].get("title", "创建新的聊天")
        return "创建新的聊天"
    
    # 定义处理流式输出的函数，确保只在流结束时更新状态
    def handle_stream_response(user_message, history, user_id, conversation_id, username):
        # 检查用户是否已登录
        if not username or not user_id or not conversation_id:
            # 返回错误提示，不清空输入框
            error_history = history + [
                {"role": "assistant", "content": "请先登录后再发送消息。"}
            ]
            yield error_history, error_history, "创建新的聊天"
            return
            
        # 检查消息是否为空
        if not user_message or not user_message.strip():
            # 返回错误提示，不清空输入框
            error_history = history + [
                {"role": "assistant", "content": "请输入有效的消息内容。"}
            ]
            # 获取当前会话标题
            current_title = None
            if username and conversation_id and username in users_db and conversation_id in users_db[username]["conversations"]:
                current_title = users_db[username]["conversations"][conversation_id].get("title", "创建新的聊天")
            yield error_history, error_history, current_title or "创建新的聊天"
            return
            
        # 获取初始会话标题
        current_title = None
        if username and conversation_id and username in users_db and conversation_id in users_db[username]["conversations"]:
            current_title = users_db[username]["conversations"][conversation_id].get("title", "创建新的聊天")
        
        # 构造发送给后端的数据，包含用户消息、用户 ID 和会话 ID
        data = {
            "messages": [{"role": "user", "content": user_message}],
            "stream": stream_flag,
            "userId": user_id,
            "conversationId": conversation_id
        }

        # 更新聊天历史，添加用户消息和临时占位回复
        history = history + [
            {"role": "user", "content": user_message},
            {"role": "assistant", "content": "正在思考中..."}
        ]
        
        # 返回初始状态，用于显示
        yield history, history, current_title
        
        # 更新为处理中的状态
        history = history[:-1] + [
            {"role": "assistant", "content": "正在生成回复..."}
        ]
        yield history, history, current_title
        
        # 如果是首次消息，设置会话标题为用户消息的前 20 个字符或完整消息
        if username and conversation_id:
            if username in users_db and conversation_id in users_db[username]["conversations"]:
                if not users_db[username]["conversations"][conversation_id].get("title_set", False):
                    new_title = user_message[:20] if len(user_message) > 20 else user_message
                    users_db[username]["conversations"][conversation_id]["title"] = new_title
                    users_db[username]["conversations"][conversation_id]["title_set"] = True
                    current_title = new_title

        # 定义格式化回复内容的函数
        def format_response(full_text):
            # 将 </think> 标签替换为加粗的"思考过程"标题
            formatted_text = re.sub(r'</think>', '**思考过程**：\n', full_text)
            # 将 </think> 标签替换为加粗的"最终回复"标题
            formatted_text = re.sub(r'</think>', '\n\n**最终回复**：\n', formatted_text)
            # 返回去除前后空白的格式化文本
            return formatted_text.strip()

        # 流式输出处理
        assistant_response = ""
        final_history = None
        content_received = False
        
        stream_finished = False
        try:
            # 添加连接状态提示
            history = history[:-1] + [
                {"role": "assistant", "content": "正在连接服务器..."}
            ]
            yield history, history, current_title
            
            with requests.post(url, headers=headers, data=json.dumps(data), stream=True, timeout=60) as response:
                # 检查响应状态
                if response.status_code != 200:
                    history = history[:-1] + [
                        {"role": "assistant", "content": f"服务器响应错误: {response.status_code}"}
                    ]
                    yield history, history, current_title
                    return
                
                # 更新为接收数据状态
                history = history[:-1] + [
                    {"role": "assistant", "content": "正在接收数据..."}
                ]
                yield history, history, current_title
                
                for line in response.iter_lines():
                    if line:
                        json_str = line.decode('utf-8').strip("data: ")
                        if not json_str:
                            logger.info("收到空字符串，跳过...")
                            continue
                        if json_str.startswith('{') and json_str.endswith('}'):
                            try:
                                response_data = json.loads(json_str)
                                # 检查response_data是否包含choices键
                                if 'choices' in response_data and len(response_data['choices']) > 0 and 'delta' in response_data['choices'][0]:
                                    content = response_data['choices'][0]['delta'].get('content', '')
                                    if content:
                                        content_received = True
                                        # 直接累积原始内容，不进行格式化
                                        logger.info(f"接收数据:{content}")
                                        assistant_response += content
                                        updated_history = history[:-1] + [
                                            {"role": "assistant", "content": assistant_response}
                                        ]
                                        # 在流式输出中yield中间状态，但不更新最终状态
                                        yield updated_history, updated_history, current_title
                                if 'choices' in response_data and len(response_data['choices']) > 0 and response_data.get('choices', [{}])[0].get('finish_reason') == "stop":
                                    logger.info("接收JSON数据结束")
                                    # 流结束时对完整响应进行最终格式化
                                    formatted_response = format_response(assistant_response)
                                    final_history = history[:-1] + [
                                        {"role": "assistant", "content": formatted_response}
                                    ]
                                    # 更新数据库中的历史记录
                                    update_history(final_history, history, username, conversation_id)
                                    yield final_history, final_history, current_title
                                    stream_finished = True
                                    break
                            except json.JSONDecodeError as e:
                                logger.error(f"JSON解析错误: {e}")
                                error_history = history[:-1] + [
                                    {"role": "assistant", "content": "解析响应时出错，请稍后再试。"}
                                ]
                                yield error_history, error_history, current_title
                                stream_finished = True
                                break
                        else:
                            logger.info(f"无效JSON格式: {json_str}")
                    else:
                        logger.info("收到空行")
                
                # 检查流是否正常结束
                if not stream_finished:
                    logger.info("流式响应结束但未明确结束")
                    if content_received:
                        # 对不完整响应进行格式化
                        formatted_response = format_response(assistant_response) if assistant_response else "未收到完整响应。"
                        final_history = history[:-1] + [
                            {"role": "assistant", "content": formatted_response}
                        ]
                        # 更新数据库中的历史记录
                        update_history(final_history, history, username, conversation_id)
                        yield final_history, final_history, current_title
                    else:
                        # 没有收到任何内容
                        error_history = history[:-1] + [
                            {"role": "assistant", "content": "未收到响应内容，请稍后再试。"}
                        ]
                        yield error_history, error_history, current_title
        except requests.Timeout:
            logger.error("请求超时")
            error_history = history[:-1] + [
                {"role": "assistant", "content": "请求超时，请稍后再试。"}
            ]
            yield error_history, error_history, current_title
        except requests.RequestException as e:
            logger.error(f"请求失败: {e}")
            error_history = history[:-1] + [
                {"role": "assistant", "content": f"请求失败: {str(e)}"}
            ]
            yield error_history, error_history, current_title

    # 绑定发送按钮点击事件，发送消息并更新界面
    send.click(
        send_message_with_state,
        [message, chatbot_history, current_user_id, current_conversation, current_user, sending_state],
        [chatbot, chatbot_history, conversation_title, sending_state, sending_indicator]
    ).then(
        # 消息发送成功后清空输入框
        lambda: "", None, message
    ).then(
        # 更新标题显示
        update_title_display, [conversation_title], title_display
    )

    # 绑定消息输入框的提交事件（Enter 键），发送消息并更新界面
    message.submit(
        send_message_with_state,
        [message, chatbot_history, current_user_id, current_conversation, current_user, sending_state],
        [chatbot, chatbot_history, conversation_title, sending_state, sending_indicator]
    ).then(
        # 消息发送成功后清空输入框
        lambda: "", None, message
    ).then(
        # 更新标题显示
        update_title_display, [conversation_title], title_display
    )

if __name__ == "__main__":
    os.environ["NO_PROXY"] = "localhost,127.0.0.1,0.0.0.0"
    os.environ["no_proxy"] = "localhost,127.0.0.1,0.0.0.0"
    for proxy_var in ["HTTP_PROXY", "HTTPS_PROXY", "ALL_PROXY", "http_proxy", "https_proxy", "all_proxy"]:
        os.environ[proxy_var] = ""
    os.environ["GRADIO_SERVER_NAME"] = "127.0.0.1"
    logger.info("Starting WebUI...")
    def pick_port(start, end):
        for port in range(start, end + 1):
            try:
                with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
                    s.bind(("127.0.0.1", port))
                return port
            except OSError:
                continue
        return None

    preferred_port = int(os.getenv("GRADIO_SERVER_PORT", "7860"))
    selected_port = pick_port(preferred_port, preferred_port) or pick_port(7861, 7870)
    if not selected_port:
        raise OSError("No available port found in range 7860-7870")
    logger.info(f"Using WebUI port: {selected_port}")
    try:
        demo.launch(server_name="127.0.0.1", server_port=selected_port, show_error=True)
    except ValueError as e:
        if "localhost is not accessible" in str(e):
            logger.warning("localhost 不可访问，尝试开启 share=True 启动")
            try:
                demo.launch(server_name="127.0.0.1", server_port=selected_port, share=True, show_error=True)
            except Exception as inner_e:
                logger.error(f"WebUI 启动失败 (重试后): {inner_e}")
                raise
        else:
            logger.error(f"WebUI 启动失败: {e}")
            raise
    except OSError as e:
        if "Cannot find empty port" in str(e):
            fallback_port = pick_port(7871, 7890)
            if not fallback_port:
                logger.error("无可用端口可回退")
                raise
            logger.warning(f"端口占用，尝试使用 {fallback_port}")
            demo.launch(server_name="127.0.0.1", server_port=fallback_port, show_error=True)
        else:
            logger.error(f"WebUI 启动失败: {e}")
            raise
    except Exception as e:
        logger.error(f"WebUI 启动过程中发生未预期的错误: {e}")
        raise
