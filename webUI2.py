# webUI.py
import gradio as gr
import os
import requests
import json
import logging
import socket
import re
import uuid
from datetime import datetime

# 设置日志的基本配置，指定日志级别为 INFO，并定义日志格式
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# 定义后端服务接口的 URL 地址
url = "http://localhost:8012/v1/chat/completions"
# 定义 HTTP 请求头，指定内容类型为 JSON
headers = {"Content-Type": "application/json"}

# 是否流式输出
stream_flag = True

# 初始化一个空字典，用于模拟用户数据库
users_db = {}
# 初始化一个空字典，用于存储用户名与用户 ID 的映射
user_id_map = {}

# 定义生成唯一用户 ID 的函数
def generate_unique_user_id(username):
    if username not in user_id_map:
        user_id = str(uuid.uuid4())
        while user_id in user_id_map.values():
            user_id = str(uuid.uuid4())
        user_id_map[username] = user_id
    return user_id_map[username]

# 定义生成唯一会话 ID 的函数
def generate_unique_conversation_id(username):
    return f"{username}_{uuid.uuid4()}"

# 定义发送消息的函数，处理用户输入并获取后端回复
def send_message(user_message, history, user_id, conversation_id, username):
    data = {
        "messages": [{"role": "user", "content": user_message}],
        "stream": stream_flag,
        "userId": user_id,
        "conversationId": conversation_id
    }

    history = history + [
        {"role": "user", "content": user_message},
        {"role": "assistant", "content": "正在生成回复..."}
    ]
    
    current_title = None
    if username and conversation_id and username in users_db and conversation_id in users_db[username]["conversations"]:
        current_title = users_db[username]["conversations"][conversation_id].get("title", "创建新的聊天")
    
    yield history, history, current_title

    if username and conversation_id:
        if username in users_db and conversation_id in users_db[username]["conversations"]:
            if not users_db[username]["conversations"][conversation_id].get("title_set", False):
                new_title = user_message[:20] if len(user_message) > 20 else user_message
                users_db[username]["conversations"][conversation_id]["title"] = new_title
                users_db[username]["conversations"][conversation_id]["title_set"] = True

    def format_response(full_text):
        formatted_text = re.sub(r'<think>', '**思考过程**：\n', full_text)
        formatted_text = re.sub(r'</think>', '\n\n**最终回复**：\n', formatted_text)
        return formatted_text.strip()

    if stream_flag:
        assistant_response = ""
        try:
            with requests.post(url, headers=headers, data=json.dumps(data), stream=True, timeout=60) as response:
                for line in response.iter_lines():
                    if line:
                        json_str = line.decode('utf-8').strip("data: ")
                        if not json_str:
                            continue
                        if json_str.startswith('{') and json_str.endswith('}'):
                            try:
                                response_data = json.loads(json_str)
                                if 'delta' in response_data['choices'][0]:
                                    content = response_data['choices'][0]['delta'].get('content', '')
                                    assistant_response += content
                                    updated_history = history[:-1] + [
                                        {"role": "assistant", "content": assistant_response}
                                    ]
                                    yield updated_history, updated_history, current_title
                                if response_data.get('choices', [{}])[0].get('finish_reason') == "stop":
                                    formatted_response = format_response(assistant_response)
                                    final_history = history[:-1] + [
                                        {"role": "assistant", "content": formatted_response}
                                    ]
                                    yield final_history, final_history, current_title
                                    break
                            except json.JSONDecodeError:
                                error_history = history[:-1] + [
                                    {"role": "assistant", "content": "解析响应时出错，请稍后再试。"}
                                ]
                                yield error_history, error_history, current_title
                                break
                else:
                    formatted_response = format_response(assistant_response) if assistant_response else "未收到完整响应。"
                    incomplete_history = history[:-1] + [
                        {"role": "assistant", "content": formatted_response}
                    ]
                    yield incomplete_history, incomplete_history, current_title
        except requests.RequestException:
            error_history = history[:-1] + [
                {"role": "assistant", "content": "请求失败，请稍后再试。"}
            ]
            yield error_history, error_history, current_title
    else:
        current_title = None
        if username and conversation_id and username in users_db and conversation_id in users_db[username]["conversations"]:
            current_title = users_db[username]["conversations"][conversation_id].get("title", "创建新的聊天")
            
        try:
            response = requests.post(url, headers=headers, data=json.dumps(data), timeout=60)
            response_json = response.json()
            assistant_content = response_json['choices'][0]['message']['content']
            formatted_content = format_response(assistant_content)
            updated_history = history[:-1] + [
                {"role": "assistant", "content": formatted_content}
            ]
            yield updated_history, updated_history, current_title
        except requests.RequestException:
            error_history = history[:-1] + [
                {"role": "assistant", "content": "请求失败，请稍后再试。"}
            ]
            yield error_history, error_history, current_title

# 定义注册用户的函数
def register(username, password):
    if username in users_db:
        return "用户名已存在！"
    user_id = generate_unique_user_id(username)
    users_db[username] = {"password": password, "user_id": user_id, "conversations": {}}
    return "注册成功！请关闭弹窗并登录。"

# 定义用户登录的函数
def login(username, password):
    if username in users_db and users_db[username]["password"] == password:
        user_id = users_db[username]["user_id"]
        conversation_id = generate_unique_conversation_id(username)
        create_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        users_db[username]["conversations"][conversation_id] = {
            "history": [],
            "title": "创建新的聊天",
            "create_time": create_time,
            "title_set": False
        }
        result = (
            True, username, user_id, conversation_id, "登录成功！",
            gr.update(visible=False), gr.update(visible=False), gr.update(visible=True), "chat"
        )
        return result
    result = (
        False, None, None, None, "用户名或密码错误！",
        gr.update(visible=True), gr.update(visible=False), gr.update(visible=False), "login"
    )
    return result

# 定义创建新会话的函数
def new_conversation(username):
    if username not in users_db:
        return "请先登录！", None
    conversation_id = generate_unique_conversation_id(username)
    create_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    users_db[username]["conversations"][conversation_id] = {
        "history": [],
        "title": "创建新的聊天",
        "create_time": create_time,
        "title_set": False
    }
    return "新会话创建成功！", conversation_id

# 定义获取会话列表的函数
def get_conversation_list(username):
    if username not in users_db or not users_db[username]["conversations"]:
        return ["请选择历史会话"]
    conv_list = []
    for conv_id, details in users_db[username]["conversations"].items():
        title = details.get("title", "未命名会话")
        create_time = details.get("create_time", datetime.now().strftime("%Y-%m-%d %H:%M:%S"))
        conv_list.append(f"{title} - {create_time}")
    return ["请选择历史会话"] + conv_list

# 定义从选项中提取会话 ID 的函数
def extract_conversation_id(selected_option, username):
    if selected_option == "请选择历史会话" or username not in users_db:
        return None
    for conv_id, details in users_db[username]["conversations"].items():
        title = details.get("title", "未命名会话")
        create_time = details.get("create_time", "")
        if f"{title} - {create_time}" == selected_option:
            return conv_id
    return None

# 定义加载会话历史的函数
def load_conversation(username, selected_option):
    if selected_option == "请选择历史会话" or username not in users_db:
        return []
    conversation_id = extract_conversation_id(selected_option, username)
    if conversation_id and conversation_id in users_db[username]["conversations"]:
        return users_db[username]["conversations"][conversation_id]["history"]
    return []

# 蓝绿炫彩主题配置
custom_theme = gr.themes.Default(
    primary_hue="teal",
    secondary_hue="cyan",
    neutral_hue="slate",
    font=[gr.themes.GoogleFont("Segoe UI"), "ui-sans-serif", "system-ui", "sans-serif"],
).set(
    body_background_fill="linear-gradient(135deg, #0a192f 0%, #022c43 50%, #11998e 100%)",
    body_text_color="#e6f1ff",
    background_fill_primary="rgba(2, 12, 27, 0.7)",
    background_fill_secondary="rgba(2, 12, 27, 0.5)",
    border_color_primary="rgba(100, 255, 218, 0.3)",
    block_background_fill="rgba(2, 12, 27, 0.6)",
    block_border_width="1px",
    block_border_color="rgba(100, 255, 218, 0.3)",
    button_primary_background_fill="linear-gradient(90deg, #11998e 0%, #38ef7d 100%)",
    button_primary_text_color="#020c1b",
    button_secondary_background_fill="transparent",
    button_secondary_border_color="#64ffda",
    button_secondary_text_color="#64ffda",
    input_background_fill="rgba(2, 12, 27, 0.8)",
    input_border_color="rgba(100, 255, 218, 0.3)",
)

custom_css = """
    /* 登录、注册、弹窗容器 - 玻璃拟态效果 */
    .login-container, .register-container, .modal {
        max-width: 400px;
        margin: 0 auto;
        padding: 30px;
        border-radius: 15px !important;
        backdrop-filter: blur(10px);
        box-shadow: 0 8px 32px 0 rgba(0, 0, 0, 0.37), 0 0 15px rgba(100, 255, 218, 0.2) !important;
    }
    
    /* 头部区域 */
    .header {
        padding: 15px;
        border-radius: 10px !important;
        margin-bottom: 15px;
        box-shadow: 0 4px 15px rgba(0, 0, 0, 0.2);
    }
    
    .header-btn {
        margin: 0 5px;
    }
    
    /* 聊天区域容器 */
    .chat-area {
        max-width: 900px;
        margin: 0 auto;
        border-radius: 15px !important;
        padding: 20px;
        box-shadow: 0 8px 32px rgba(0, 0, 0, 0.3);
    }
    
    /* 弹窗居中 */
    .modal {
        position: fixed;
        top: 50%;
        left: 50%;
        transform: translate(-50%, -50%);
        z-index: 1000;
        width: 90%;
    }
    
    /* 切换提示文本 */
    .switch-prompt {
        text-align: center;
        margin: 15px 0;
        color: #8892b0 !important;
    }
    
    /* 发送状态指示器 - 炫彩呼吸灯效果 */
    .sending-indicator {
        display: inline-block;
        padding: 8px 15px;
        background: linear-gradient(90deg, rgba(17, 153, 142, 0.2), rgba(56, 239, 125, 0.2));
        color: #64ffda;
        border-radius: 8px;
        font-size: 14px;
        margin-top: 5px;
        border: 1px solid rgba(100, 255, 218, 0.4);
        animation: neon-pulse 1.5s infinite alternate;
    }
    
    @keyframes neon-pulse {
        from { box-shadow: 0 0 5px rgba(100, 255, 218, 0.2); }
        to { box-shadow: 0 0 15px rgba(100, 255, 218, 0.6), 0 0 20px rgba(56, 239, 125, 0.4); }
    }
    
    #title-display {
        text-align: center;
        margin-bottom: 15px;
        color: #64ffda !important;
        text-shadow: 0 0 10px rgba(100, 255, 218, 0.3);
    }
    
    /* 按钮悬浮效果 */
    button.primary:hover {
        transform: translateY(-2px);
        box-shadow: 0 6px 20px rgba(56, 239, 125, 0.6) !important;
    }
    
    button.secondary:hover {
        background: rgba(100, 255, 218, 0.1) !important;
        box-shadow: 0 0 10px rgba(100, 255, 218, 0.3) !important;
    }
    
    /* 输入框焦点效果 */
    input:focus, textarea:focus {
        box-shadow: 0 0 10px rgba(100, 255, 218, 0.2) !important;
        outline: none !important;
    }
"""

# 使用 Gradio Blocks 创建前端界面
with gr.Blocks(title="聊天助手 - 蓝绿炫彩版", theme=custom_theme, css=custom_css) as demo:
    logged_in = gr.State(False)
    current_user = gr.State(None)
    current_user_id = gr.State(None)
    sending_state = gr.State(False)
    
    def send_message_with_state(user_message, chat_history, user_id, conversation_id, username, is_sending):
        if is_sending:
            yield chat_history, chat_history, "创建新的聊天", True, gr.update(visible=True, value="<div class='sending-indicator'>正在发送中...</div>")
            return
        
        yield chat_history, chat_history, "创建新的聊天", True, gr.update(visible=True, value="<div class='sending-indicator'>正在发送中...</div>")
        
        for response in handle_stream_response(user_message, chat_history, user_id, conversation_id, username):
            chatbot, chatbot_history, title = response
            yield chatbot, chatbot_history, title, True, gr.update(visible=True, value="<div class='sending-indicator'>正在接收回复...</div>")
        
        yield chatbot, chatbot_history, title, False, gr.update(visible=False)
        
    current_conversation = gr.State(None)
    chatbot_history = gr.State([])
    conversation_title = gr.State("创建新的聊天")
    page_state = gr.State("login")
    
    with gr.Column(visible=True, elem_classes="login-container") as login_page:
        gr.Markdown("## 聊天助手")
        login_username = gr.Textbox(label="用户名", placeholder="请输入用户名")
        login_password = gr.Textbox(label="密码", placeholder="请输入密码", type="password")
        with gr.Row():
            login_button = gr.Button("登录", variant="primary")
            register_button = gr.Button("注册", variant="secondary")
        login_output = gr.Textbox(label="结果", interactive=False)
        
        gr.Markdown("还没有账号？", elem_classes="switch-prompt")
        switch_to_register = gr.Button("立即注册", variant="secondary", size="sm")

    with gr.Column(visible=False, elem_classes="register-container") as register_page:
        gr.Markdown("## 注册新账号")
        reg_username = gr.Textbox(label="用户名", placeholder="请输入用户名")
        reg_password = gr.Textbox(label="密码", placeholder="请输入密码", type="password")
        with gr.Row():
            reg_button = gr.Button("提交注册", variant="primary")
            switch_to_login = gr.Button("返回登录", variant="secondary", size="sm")
        reg_output = gr.Textbox(label="结果", interactive=False)
        
        gr.Markdown("已有账号？", elem_classes="switch-prompt")
        switch_to_login_from_register = gr.Button("立即登录", variant="secondary", size="sm")

    with gr.Column(visible=False) as chat_page:
        with gr.Row(elem_classes="header"):
            welcome_text = gr.Markdown("### 欢迎，")
            with gr.Row():
                new_conv_button = gr.Button("新建会话", elem_classes="header-btn", variant="secondary")
                history_button = gr.Button("历史会话", elem_classes="header-btn", variant="secondary")
                logout_button = gr.Button("退出登录", elem_classes="header-btn", variant="secondary")

        with gr.Column(elem_classes="chat-area"):
            title_display = gr.Markdown("## 会话标题", elem_id="title-display")
            chatbot = gr.Chatbot(label="聊天对话", height=450, type="messages")
            with gr.Row():
                message = gr.Textbox(label="消息", placeholder="输入消息并按 Enter 发送", scale=8, container=False)
                send = gr.Button("发送", scale=2, variant="primary")
                sending_indicator = gr.HTML("", visible=False, elem_id="sending-indicator")

    with gr.Column(visible=False, elem_classes="modal") as history_modal:
        gr.Markdown("### 会话历史")
        conv_dropdown = gr.Dropdown(label="选择历史会话", choices=["请选择历史会话"], value="请选择历史会话")
        load_conv_button = gr.Button("加载会话", variant="primary")
        close_history_button = gr.Button("关闭", variant="secondary")

    def show_login_page():
        return gr.update(visible=True), gr.update(visible=False), gr.update(visible=False), "login"
    
    def show_register_page():
        return gr.update(visible=False), gr.update(visible=True), gr.update(visible=False), "register"
    
    def show_chat_page():
        return gr.update(visible=False), gr.update(visible=False), gr.update(visible=True), "chat"

    def show_history_modal(username):
        return gr.update(visible=True), gr.update(choices=get_conversation_list(username), value="请选择历史会话")

    def hide_history_modal():
        return gr.update(visible=False)

    def logout():
        return (
            False, None, None,
            gr.update(visible=True), gr.update(visible=False), gr.update(visible=False), "login",
            "已退出登录", [], None, [], "创建新的聊天",
            gr.update(value=""), gr.update(value="")
        )

    def update_welcome_text(username):
        return gr.update(value=f"### 欢迎，{username}")

    def update_title_display(title):
        return gr.update(value=f"## {title}")

    register_button.click(show_register_page, None, [login_page, register_page, chat_page, page_state])
    switch_to_register.click(show_register_page, None, [login_page, register_page, chat_page, page_state])
    switch_to_login.click(show_login_page, None, [login_page, register_page, chat_page, page_state])
    switch_to_login_from_register.click(show_login_page, None, [login_page, register_page, chat_page, page_state])
    
    reg_button.click(
        register, [reg_username, reg_password], reg_output
    ).then(
        lambda result: [gr.update(value=""), gr.update(value="")] if result and "注册成功" in result else [gr.update(value=""), gr.update(value="")],
        None,
        [reg_username, reg_password]
    ).then(
        lambda result: show_login_page() if result and "注册成功" in result else (gr.update(visible=True), gr.update(visible=False), gr.update(visible=False), "login"),
        None,
        [login_page, register_page, chat_page, page_state]
    )

    login_button.click(
        login, [login_username, login_password],
        [logged_in, current_user, current_user_id, current_conversation, login_output,
         login_page, register_page, chat_page, page_state]
    ).then(
        lambda logged_in, username: update_welcome_text(username) if logged_in else gr.update(value="### 欢迎，访客"),
        [logged_in, current_user], 
        welcome_text
    ).then(
        lambda logged_in, username, conv_id: users_db[username]["conversations"][conv_id]["history"]
        if logged_in and username and conv_id and username in users_db and conv_id in users_db.get(username, {}).get("conversations", {})
        else [],
        [logged_in, current_user, current_conversation], 
        chatbot_history
    ).then(
        lambda logged_in, username, conv_id: users_db[username]["conversations"][conv_id].get("title", "创建新的聊天")
        if logged_in and username and conv_id and username in users_db and conv_id in users_db.get(username, {}).get("conversations", {})
        else "创建新的聊天",
        [logged_in, current_user, current_conversation], 
        conversation_title
    ).then(
        update_title_display, 
        [conversation_title], 
        title_display
    )

    logout_button.click(
        logout, None,
        [logged_in, current_user, current_user_id, login_page, register_page, chat_page, page_state,
         login_output, chatbot, current_conversation, chatbot_history, conversation_title,
         login_username, login_password]
    )

    history_button.click(show_history_modal, [current_user], [history_modal, conv_dropdown])
    close_history_button.click(hide_history_modal, None, history_modal)

    new_conv_button.click(
        new_conversation, [current_user], [current_conversation]
    ).then(
        lambda: [], None, chatbot
    ).then(
        lambda: [], None, chatbot_history
    ).then(
        lambda: "创建新的聊天", None, conversation_title
    ).then(
        update_title_display, [conversation_title], title_display
    )

    load_conv_button.click(
        load_conversation, [current_user, conv_dropdown], [chatbot, chatbot_history]
    ).then(
        lambda user, conv: extract_conversation_id(conv, user),
        [current_user, conv_dropdown], current_conversation
    ).then(
        lambda username, conv: users_db[username]["conversations"][
            extract_conversation_id(conv, username)
        ].get("title", "创建新的聊天")
        if username and conv and username in users_db and extract_conversation_id(conv, username)
        else "创建新的聊天",
        [current_user, conv_dropdown], conversation_title
    ).then(
        update_title_display, [conversation_title], title_display
    ).then(
        hide_history_modal, None, history_modal
    )

    def update_history(chatbot_output, history, user, conv_id):
        if user and conv_id and user in users_db and conv_id in users_db.get(user, {}).get("conversations", {}):
            users_db[user]["conversations"][conv_id]["history"] = chatbot_output
        return chatbot_output

    def safe_get_title(username, conv_id):
        if (username and conv_id and username in users_db
                and conv_id in users_db.get(username, {}).get("conversations", {})):
            return users_db[username]["conversations"][conv_id].get("title", "创建新的聊天")
        return "创建新的聊天"
    
    def handle_stream_response(user_message, history, user_id, conversation_id, username):
        if not username or not user_id or not conversation_id:
            error_history = history + [
                {"role": "assistant", "content": "请先登录后再发送消息。"}
            ]
            yield error_history, error_history, "创建新的聊天"
            return
            
        if not user_message or not user_message.strip():
            error_history = history + [
                {"role": "assistant", "content": "请输入有效的消息内容。"}
            ]
            current_title = None
            if username and conversation_id and username in users_db and conversation_id in users_db[username]["conversations"]:
                current_title = users_db[username]["conversations"][conversation_id].get("title", "创建新的聊天")
            yield error_history, error_history, current_title or "创建新的聊天"
            return
            
        current_title = None
        if username and conversation_id and username in users_db and conversation_id in users_db[username]["conversations"]:
            current_title = users_db[username]["conversations"][conversation_id].get("title", "创建新的聊天")
        
        data = {
            "messages": [{"role": "user", "content": user_message}],
            "stream": stream_flag,
            "userId": user_id,
            "conversationId": conversation_id
        }

        history = history + [
            {"role": "user", "content": user_message},
            {"role": "assistant", "content": "正在思考中..."}
        ]
        
        yield history, history, current_title
        
        history = history[:-1] + [
            {"role": "assistant", "content": "正在生成回复..."}
        ]
        yield history, history, current_title
        
        if username and conversation_id:
            if username in users_db and conversation_id in users_db[username]["conversations"]:
                if not users_db[username]["conversations"][conversation_id].get("title_set", False):
                    new_title = user_message[:20] if len(user_message) > 20 else user_message
                    users_db[username]["conversations"][conversation_id]["title"] = new_title
                    users_db[username]["conversations"][conversation_id]["title_set"] = True
                    current_title = new_title

        def format_response(full_text):
            formatted_text = re.sub(r'</think>', '**思考过程**：\n', full_text)
            formatted_text = re.sub(r'</think>', '\n\n**最终回复**：\n', formatted_text)
            return formatted_text.strip()

        assistant_response = ""
        final_history = None
        content_received = False
        
        stream_finished = False
        try:
            history = history[:-1] + [
                {"role": "assistant", "content": "正在连接服务器..."}
            ]
            yield history, history, current_title
            
            with requests.post(url, headers=headers, data=json.dumps(data), stream=True, timeout=60) as response:
                if response.status_code != 200:
                    history = history[:-1] + [
                        {"role": "assistant", "content": f"服务器响应错误: {response.status_code}"}
                    ]
                    yield history, history, current_title
                    return
                
                history = history[:-1] + [
                    {"role": "assistant", "content": "正在接收数据..."}
                ]
                yield history, history, current_title
                
                for line in response.iter_lines():
                    if line:
                        json_str = line.decode('utf-8').strip("data: ")
                        if not json_str:
                            continue
                        if json_str.startswith('{') and json_str.endswith('}'):
                            try:
                                response_data = json.loads(json_str)
                                if 'choices' in response_data and len(response_data['choices']) > 0 and 'delta' in response_data['choices'][0]:
                                    content = response_data['choices'][0]['delta'].get('content', '')
                                    if content:
                                        content_received = True
                                        assistant_response += content
                                        updated_history = history[:-1] + [
                                            {"role": "assistant", "content": assistant_response}
                                        ]
                                        yield updated_history, updated_history, current_title
                                if 'choices' in response_data and len(response_data['choices']) > 0 and response_data.get('choices', [{}])[0].get('finish_reason') == "stop":
                                    formatted_response = format_response(assistant_response)
                                    final_history = history[:-1] + [
                                        {"role": "assistant", "content": formatted_response}
                                    ]
                                    update_history(final_history, history, username, conversation_id)
                                    yield final_history, final_history, current_title
                                    stream_finished = True
                                    break
                            except json.JSONDecodeError:
                                error_history = history[:-1] + [
                                    {"role": "assistant", "content": "解析响应时出错，请稍后再试。"}
                                ]
                                yield error_history, error_history, current_title
                                stream_finished = True
                                break
                
                if not stream_finished:
                    if content_received:
                        formatted_response = format_response(assistant_response) if assistant_response else "未收到完整响应。"
                        final_history = history[:-1] + [
                            {"role": "assistant", "content": formatted_response}
                        ]
                        update_history(final_history, history, username, conversation_id)
                        yield final_history, final_history, current_title
                    else:
                        error_history = history[:-1] + [
                            {"role": "assistant", "content": "未收到响应内容，请稍后再试。"}
                        ]
                        yield error_history, error_history, current_title
        except requests.Timeout:
            error_history = history[:-1] + [
                {"role": "assistant", "content": "请求超时，请稍后再试。"}
            ]
            yield error_history, error_history, current_title
        except requests.RequestException as e:
            error_history = history[:-1] + [
                {"role": "assistant", "content": f"请求失败: {str(e)}"}
            ]
            yield error_history, error_history, current_title

    send.click(
        send_message_with_state,
        [message, chatbot_history, current_user_id, current_conversation, current_user, sending_state],
        [chatbot, chatbot_history, conversation_title, sending_state, sending_indicator]
    ).then(
        lambda: "", None, message
    ).then(
        update_title_display, [conversation_title], title_display
    )

    message.submit(
        send_message_with_state,
        [message, chatbot_history, current_user_id, current_conversation, current_user, sending_state],
        [chatbot, chatbot_history, conversation_title, sending_state, sending_indicator]
    ).then(
        lambda: "", None, message
    ).then(
        update_title_display, [conversation_title], title_display
    )

if __name__ == "__main__":
    os.environ["NO_PROXY"] = "localhost,127.0.0.1,0.0.0.0"
    os.environ["no_proxy"] = "localhost,127.0.0.1,0.0.0.0"
    for proxy_var in ["HTTP_PROXY", "HTTPS_PROXY", "ALL_PROXY", "http_proxy", "https_proxy", "all_proxy"]:
        os.environ[proxy_var] = ""
    os.environ["GRADIO_SERVER_NAME"] = "0.0.0.0"
    logger.info("Starting WebUI...")
    def pick_port(start, end):
        for port in range(start, end + 1):
            try:
                with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
                    s.bind(("0.0.0.0", port))
                return port
            except OSError:
                continue
        return None

    preferred_port = int(os.getenv("PORT", "3000"))
    selected_port = pick_port(preferred_port, preferred_port) or pick_port(7861, 7870)
    if not selected_port:
        raise OSError("No available port found")
    logger.info(f"Using WebUI port: {selected_port}")
    try:
        demo.launch(server_name="0.0.0.0", server_port=selected_port, show_error=True)
    except Exception as e:
        logger.error(f"WebUI 启动过程中发生未预期的错误: {e}")
        raise
