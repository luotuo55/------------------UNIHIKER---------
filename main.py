from flask import Flask, request, send_file, jsonify
from difflib import SequenceMatcher
import json
import uuid
import gzip
import asyncio
import websockets
import numpy as np
import sounddevice as sd
import nest_asyncio
from unihiker import GUI
import time
from tkinter import END
import requests
import os
import dashscope
import threading
import socket
import psutil
import sys

# Flask 应用
app = Flask(__name__)

# 共享数据存储
class SharedData:
    def __init__(self):
        self.speech_recognition_results = []
        self.latest_text = ""
        self.target_text = ""
        self.latest_score = None

shared_data = SharedData()

# Web 服务器路由
@app.route('/')
def index():
    return send_file('index.html')

@app.route('/submit', methods=['POST'])
def submit():
    try:
        print("收到提交请求")  # 调试信息
        data = request.get_json()
        print(f"接收到的数据: {data}")  # 调试信息
        
        if not data:
            print("未接收到数据")
            return jsonify({"status": "error", "message": "未接收到数据"}), 400
            
        text = data.get('text', '').strip()
        print(f"提取的文本: {text}")  # 调试信息
        
        if not text:
            print("文本为空")
            return jsonify({"status": "error", "message": "范文不能为空"}), 400
            
        shared_data.target_text = text
        print(f"已保存范文: {shared_data.target_text}")  # 调试信息
        
        return jsonify({
            "status": "success", 
            "message": "范文提交成功",
            "text_length": len(text)
        })
        
    except Exception as e:
        print(f"提交处理错误: {str(e)}")  # 调试信息
        return jsonify({"status": "error", "message": str(e)}), 500

@app.route('/get_score')
def get_score():
    if shared_data.latest_score:
        score_data = shared_data.latest_score
        # 返回后清除分数，避免重复获取
        shared_data.latest_score = None
        return jsonify({"status": "success", "data": score_data})
    return jsonify({"status": "waiting"})

# 相似度计算函数
def calculate_similarity(text1, text2):
    return SequenceMatcher(None, text1, text2).ratio()

def calculate_completeness(target_text, speech_text):
    target_words = set(target_text.split())
    speech_words = set(speech_text.split())
    common_words = target_words.intersection(speech_words)
    return len(common_words) / len(target_words) if target_words else 0

def get_similarity_comment(similarity):
    if similarity >= 0.9: return "非常接近"
    elif similarity >= 0.7: return "比较接近"
    elif similarity >= 0.5: return "部分接近"
    else: return "差异较大"

def get_completeness_comment(completeness):
    if completeness >= 0.9: return "内容完整"
    elif completeness >= 0.7: return "大部分完整"
    elif completeness >= 0.5: return "部分完整"
    else: return "内容缺失"

# 配置参数
appid = "4166554764"    # 项目的 appid
token = "ggmUTHHMXio-nJlKMkRvqEgkcWyfDK0K"    # 项目的 token
cluster = "volcengine_streaming_common"  # 请求的集群

# 协议常量
PROTOCOL_VERSION = 0b0001
DEFAULT_HEADER_SIZE = 0b0001

PROTOCOL_VERSION_BITS = 4
HEADER_BITS = 4
MESSAGE_TYPE_BITS = 4
MESSAGE_TYPE_SPECIFIC_FLAGS_BITS = 4
MESSAGE_SERIALIZATION_BITS = 4
MESSAGE_COMPRESSION_BITS = 4
RESERVED_BITS = 8

# Message Type:
CLIENT_FULL_REQUEST = 0b0001
CLIENT_AUDIO_ONLY_REQUEST = 0b0010
SERVER_FULL_RESPONSE = 0b1001
SERVER_ACK = 0b1011
SERVER_ERROR_RESPONSE = 0b1111

# Message Type Specific Flags
NO_SEQUENCE = 0b0000
POS_SEQUENCE = 0b0001
NEG_SEQUENCE = 0b0010
NEG_SEQUENCE_1 = 0b0011

# Message Serialization
NO_SERIALIZATION = 0b0000
JSON = 0b0001
THRIFT = 0b0011
CUSTOM_TYPE = 0b1111

# Message Compression
NO_COMPRESSION = 0b0000
GZIP = 0b0001
CUSTOM_COMPRESSION = 0b1111

# 初始化 nest_asyncio
nest_asyncio.apply()

# 全局变量
is_recording = True
all_texts = []
gui = None
loop = None  # 添加全局 loop 变量

# 在程序最开始处（所有 import 语句之后）设置 API key
def init_api_keys():
    """初始化API密钥"""
    try:
        # 语音识别配置
        global appid, token, cluster
        appid = "4166554764"
        token = "ggmUTHHMXio-nJlKMkRvqEgkcWyfDK0K"
        cluster = "volcengine_streaming_common"
        
        # 千问API配置
        dashscope_key = 'sk-7c04ee6f9432492bb344baa7a5c0162f'
        os.environ['DASHSCOPE_API_KEY'] = dashscope_key
        dashscope.api_key = dashscope_key
        
        # 验证千问API key是否设置成功
        print(f"当前 DashScope API Key: {dashscope.api_key}")
        return True
        
    except Exception as e:
        print(f"API密钥初始化失败: {e}")
        return False

def call_qwen_api(text):
    """调用千问API进行文本纠错"""
    try:
        if not dashscope.api_key:
            print("[错误] DashScope API Key未设置")
            return None
            
        print(f"使用的API Key: {dashscope.api_key}")
        
        messages = [
            {
                'role': 'system',
                'content': '你是一个专业的文本纠错助手。请对输入的文本进行标点符号和错别字的修正，保持原文的意思不变。'
            },
            {
                'role': 'user',
                'content': f'请修正以下文本的标点符号和错别字：\n{text}'
            }
        ]
        
        response = dashscope.Generation.call(
            model='qwen-plus',
            messages=messages,
            result_format='message',
            api_key=dashscope.api_key  # 显式传递API key
        )
        
        if response.status_code == 200:
            corrected_text = response.output.choices[0].message.content.strip()
            print(f"[纠正结果] {corrected_text}")
            return corrected_text
        else:
            error_msg = f"API调用失败: {response.code} - {response.message}"
            print(f"[错误] {error_msg}")
            return None
            
    except Exception as e:
        error_msg = f"纠错API调用失败: {str(e)}"
        print(f"[错误] {error_msg}")
        return None

def on_correction_click():
    """处理纠错按钮点击事件"""
    global is_recording
    try:
        print("纠错按钮被点击")
        
        # 停止录音
        is_recording = False
        print("正在停止录音...")
        time.sleep(1)
        
        # 获取当前文本
        current_text = text_box.text.get("1.0", END)
        print(f"正在纠错文本: {current_text}")
        
        if not current_text.strip():
            print("文本为空，无法纠错")
            update_recognition_text("请先进行语音识别")
            return
            
        print("\n开始文本纠错...")
        
        # 构造提示词
        prompt = f"""
请对以下文本进行语法和用词纠错。
如果发现错误，请返回修正后的完整文本；如果没有错误，请返回原文。

文本内容：
{current_text}

请直接返回纠错后的文本，不需要其他解释。
"""
        
        # 调用千问API
        response = dashscope.Generation.call(
            model='qwen-plus',
            messages=[
                {'role': 'system', 'content': '你是一个专业的文本纠错助手'},
                {'role': 'user', 'content': prompt}
            ],
            result_format='message',
            api_key=dashscope.api_key
        )
        
        print(f"API响应状态码: {response.status_code}")
        print(f"API完整响应: {response}")
        
        if response.status_code == 200:
            corrected_text = response.output.choices[0].message.content.strip()
            print(f"API返回结果: {corrected_text}")
            
            # 添加【纠错结果】标识并追加显示
            display_text = f"\n【纠错结果】\n{corrected_text}"
            update_recognition_text(display_text)
            print("纠错完成")
            
        else:
            error_msg = f"API调用失败: {response.message}"
            print(error_msg)
            update_recognition_text(error_msg)
            
    except Exception as e:
        error_msg = f"文本纠错错误: {str(e)}"
        print(error_msg)
        update_recognition_text(error_msg)

def on_score_click():
    """处理打分按钮点击事件"""
    global is_recording
    try:
        print("打分按钮被点击")
        
        # 停止语音识别
        is_recording = False
        print("正在停止语...")
        time.sleep(1)
        
        # 获取当前文本
        current_text = text_box.text.get("1.0", END)
        print(f"获取到的朗读文本：{current_text}")
        
        # 获取范文
        target_text = shared_data.target_text
        if not target_text:
            error_msg = "[错误] 请先在网页端输入范文"
            print(error_msg)
            update_recognition_text(error_msg)
            return
            
        if not current_text.strip():
            error_msg = "[错误] 请先进行语音识别"
            print(error_msg)
            update_recognition_text(error_msg)
            return
        
        print("开始调用API进行评分...")
        # 构造提示词
        prompt = f"""
请对比以下两段文本，从准确度、完整度、流畅度三维度进行评分（满100分），并给出详细分析：

范文：
{target_text}

朗读文本：
{current_text}

请按以下格式输出：
准确度：XX分
完整度：XX分
流畅度：XX分
总分：XX分

详细分析：
1. 准确度分析：...
2. 完整度分析：...
3. 流畅度分析：...
4. 改进建议：...
"""
        
        # 调用API进行评分
        response = dashscope.Generation.call(
            api_key=os.getenv('DASHSCOPE_API_KEY'),
            model="qwen-plus",
            messages=[
                {'role': 'system', 'content': 'You are a helpful assistant.'},
                {'role': 'user', 'content': prompt}
            ],
            result_format='message'
        )
        
        if response.status_code == 200:
            result = response.output.choices[0].message.content
            print(f"评分结果：{result}")
            
            # 更新UNIHIKER显示
            update_recognition_text(f"[评分结果]\n{result}")
            
            # 更新共享数据供网页显示
            shared_data.latest_score = {
                "target_text": target_text,
                "speech_text": current_text,
                "score_result": result
            }
            
        else:
            error_msg = f"评分失败: {response.message}"
            print(error_msg)
            update_recognition_text(f"[错误] {error_msg}")
            
    except Exception as e:
        error_msg = f"评分错误: {str(e)}"
        print(error_msg)
        update_recognition_text(error_msg)

# 添加全局变量
recognized_name = None  # 存储识别出的姓名
is_recording = False

def init_main_gui():
    """初始化主界面GUI"""
    global gui, text_box, start_button, correction_button, score_button
    global name_button, retry_button, confirm_button
    
    if gui is None:
        gui = GUI()
    else:
        gui.clear()
    
    # 创建文本框
    text_box = gui.add_text_box(
        x=120,
        y=90,
        w=220,
        h=220,
        origin='center',
        font_size=12
    )
    
    # 计算按钮位置
    screen_height = 320
    button_height = 36
    margin_bottom = 10
    button_width = 70
    
    row2_y = screen_height - margin_bottom - button_height
    row1_y = row2_y - button_height - 10
    
    # 第一行按钮（开始、纠错、打分）
    start_button = gui.add_button(
        x=45,
        y=row1_y,
        w=button_width,
        h=button_height,
        text="开始",
        origin='center',
        onclick=start_button_click
    )
    
    correction_button = gui.add_button(
        x=120,
        y=row1_y,
        w=button_width,
        h=button_height,
        text="纠错",
        origin='center',
        onclick=on_correction_click
    )
    
    score_button = gui.add_button(
        x=195,
        y=row1_y,
        w=button_width,
        h=button_height,
        text="打分",
        origin='center',
        onclick=on_score_click
    )
    
    # 第二行按钮（姓名、重试、确认）
    name_button = gui.add_button(
        x=45,
        y=row2_y,
        w=button_width,
        h=button_height,
        text="姓名",
        origin='center',
        onclick=on_name_click
    )
    
    retry_button = gui.add_button(
        x=120,
        y=row2_y,
        w=button_width,
        h=button_height,
        text="重试",
        origin='center',
        onclick=on_retry_name
    )
    
    confirm_button = gui.add_button(
        x=195,
        y=row2_y,
        w=button_width,
        h=button_height,
        text="确认",
        origin='center',
        onclick=on_confirm_name
    )
    
    return gui, text_box

def toggle_confirmation_buttons(show_confirm=True):
    """切换确认按钮的显示状态"""
    global name_button, retry_button, confirm_button
    try:
        if show_confirm:
            # 隐藏姓名按钮，显示重试和确认按钮
            name_button.widget.pack_forget()
            retry_button.widget.pack()
            confirm_button.widget.pack()
        else:
            # 显示姓名按钮，隐藏重试和确认按钮
            name_button.widget.pack()
            retry_button.widget.pack_forget()
            confirm_button.widget.pack_forget()
            
        # 刷新GUI
        if gui and hasattr(gui, 'master'):
            gui.master.update()
            
    except Exception as e:
        print(f"切换按钮显示状态时出错: {e}")

def on_name_click():
    """处理姓名按钮点击事件"""
    global recognized_name, is_recording
    try:
        print("姓名按钮被点击")
        
        # 停止录音
        is_recording = False
        print("正在停止录音...")
        time.sleep(1)
        
        # 获取当前文本
        current_text = text_box.text.get("1.0", END)
        print(f"正在识别文本: {current_text}")
        
        if not current_text.strip():
            print("文本为空，无法识别姓名")
            update_recognition_text("请先进行语音识别")
            return
            
        print("\n开始识别姓名...")
        
        # 构造提示词
        prompt = f"""
请从以下文本中识别出人名（如果有的话）。
如果能识别出人名，只需返回这个人名；如果没有识别出人名，请返回"未识别到姓名"。

文本内容：
{current_text}

请直接返回人名或"未识别到姓名"，不需要其他解释。
"""
        
        # 调用千问API
        response = dashscope.Generation.call(
            model='qwen-plus',
            messages=[
                {'role': 'system', 'content': '你是一个专业的姓名识别助手'},
                {'role': 'user', 'content': prompt}
            ],
            result_format='message',
            api_key=dashscope.api_key
        )
        
        print(f"API响应状态码: {response.status_code}")
        print(f"API完整响应: {response}")
        
        if response.status_code == 200:
            name = response.output.choices[0].message.content.strip()
            print(f"API返回结果: {name}")
            
            if name != "未识别到姓名":
                recognized_name = name
                print(f"识别成功: {recognized_name}")
                update_recognition_text(f"识别到的姓名：{recognized_name}\n请点击确认或重试")
            else:
                print("未识别到姓名")
                update_recognition_text("未能识别出姓名，请重试")
        else:
            error_msg = f"API调用失败: {response.message}"
            print(error_msg)
            update_recognition_text(error_msg)
            
    except Exception as e:
        error_msg = f"姓名识别错误: {str(e)}"
        print(error_msg)
        update_recognition_text(error_msg)

def start_button_click():
    """开始按钮点击处理函数"""
    global loop
    print("开始按钮被点击")
    
    try:
        # 创建客户端实例
        client = AsrWsClient(appid, token, cluster)
        
        # 创建事件循环
        if loop is None or loop.is_closed():
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
        
        # 启动语音识别
        print("语音识别已启动")
        recognition_task = loop.create_task(client.process_microphone())
        
    except Exception as e:
        print(f"启动语音识别失败: {e}")

def kill_existing_flask():
    """杀死已存在的Flask进程"""
    try:
        current_pid = os.getpid()
        
        for proc in psutil.process_iter(['pid', 'name', 'connections']):
            try:
                # 跳过当前进程
                if proc.pid == current_pid:
                    continue
                    
                for conn in proc.connections():
                    if conn.laddr.port == 5000:
                        print(f"发现端口5000被进程占用 (PID: {proc.pid})")
                        proc.kill()
                        print(f"已终止进程 {proc.pid}")
                        time.sleep(1)  # 等待进程完全终止
            except (psutil.NoSuchProcess, psutil.AccessDenied):
                continue
    except Exception as e:
        print(f"清理进程时出错: {e}")

def run_flask():
    """运行Flask服务器"""
    try:
        # 先尝试释放端口
        try:
            import socket
            sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
            sock.bind(('0.0.0.0', 5000))
            sock.close()
        except Exception as e:
            print(f"端口绑定测试失败: {e}")
            # 尝试结束占用端口的进程
            try:
                import os
                os.system("fuser -k 5000/tcp")
                print("已尝试释放端口5000")
                time.sleep(2)
            except:
                print("无法释放端口，请手动结束占用端口的进程")
                return False

        ip = get_ip_address()
        server_info = """
╔════════════════════════════════════════════════╗
║             Web服务器启动信息                  ║
╠════════════════════════════════════════════════╣
║                                                ║
║  状态: 正在启动...                            ║
║  访问地址: http://{ip}:5000                   ║
║  本地地址: http://localhost:5000              ║
║  监听端口: 5000                               ║
║                                                ║
║  请在浏览器中访问以上地址来输入范文          ║
║                                                ║
╚════════════════════════════════════════════════╝
""".format(ip=ip)
        
        print("\n" + server_info, flush=True)
        
        app.run(host='0.0.0.0', port=5000, debug=False, use_reloader=False)
        return True
    except Exception as e:
        print("\n" + "="*50)
        print("【服务器启动失败】")
        print(f"错误信息: {str(e)}")
        print("="*50 + "\n")
        return False

def get_ip_address():
    """获取本机IP地址"""
    try:
        s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        s.connect(("8.8.8.8", 80))
        ip = s.getsockname()[0]
        s.close()
        return ip
    except:
        return "127.0.0.1"

class AsrWsClient:
    def __init__(self, appid, token, cluster):
        self.appid = appid
        self.token = token
        self.cluster = cluster
        self.ws_url = "wss://openspeech.bytedance.com/api/v2/asr"
        self.success_code = 1000
        self.uid = "streaming_asr_demo"
        self.workflow = "audio_in,resample,partition,vad,fe,decode,itn,nlu_punctuate"
        self.show_language = False
        self.show_utterances = True
        self.result_type = "single"
        self.format = "raw"
        self.rate = 16000
        self.language = "zh-CN"
        self.bits = 16
        self.channel = 1
        self.codec = "raw"
        self.auth_method = "token"

    def construct_request(self, reqid):
        return {
            'app': {
                'appid': self.appid,
                'cluster': self.cluster,
                'token': self.token,
            },
            'user': {
                'uid': self.uid
            },
            'request': {
                'reqid': reqid,
                'nbest': 1,
                'workflow': self.workflow,
                'show_language': self.show_language,
                'show_utterances': self.show_utterances,
                'result_type': self.result_type,
                'sequence': 1
            },
            'audio': {
                'format': self.format,
                'rate': self.rate,
                'language': self.language,
                'bits': self.bits,
                'channel': self.channel,
                'codec': self.codec
            }
        }

    def token_auth(self):
        return {'Authorization': f'Bearer; {self.token}'}

    async def process_microphone(self):
        """实时麦克风录音并识别"""
        global is_recording
        is_recording = True
        
        reqid = str(uuid.uuid4())
        request_params = self.construct_request(reqid)
        
        payload_bytes = str.encode(json.dumps(request_params))
        payload_bytes = gzip.compress(payload_bytes)
        full_request = bytearray(generate_full_default_header())
        full_request.extend(len(payload_bytes).to_bytes(4, 'big'))
        full_request.extend(payload_bytes)

        print("建立WebSocket连接...")
        async with websockets.connect(
            self.ws_url, 
            extra_headers=self.token_auth(), 
            max_size=1000000000
        ) as ws:
            await ws.send(full_request)
            response = await ws.recv()
            result = parse_response(response)
            print(f"初始化响应: {result}")
            
            if 'payload_msg' in result and result['payload_msg']['code'] == self.success_code:
                print("初始化成功")
                print("录音任务已启动")
                chunk_size = 9600
                
                with sd.InputStream(channels=1, samplerate=16000, dtype=np.int16, blocksize=chunk_size) as stream:
                    print("开始录音...")
                    while is_recording:
                        audio_data, overflowed = stream.read(chunk_size)
                        if overflowed:
                            print("警告：音频缓冲区溢出")
                            
                        audio_bytes = audio_data.tobytes()
                        compressed_audio = gzip.compress(audio_bytes)
                        
                        audio_request = bytearray(generate_audio_default_header())
                        audio_request.extend(len(compressed_audio).to_bytes(4, 'big'))
                        audio_request.extend(compressed_audio)
                        
                        await ws.send(audio_request)
                        response = await ws.recv()
                        result = parse_response(response)
                        
                        if 'payload_msg' in result and 'result' in result['payload_msg']:
                            utterances = result['payload_msg']['result'][0].get('utterances', [])
                            for utterance in utterances:
                                if not utterance['definite']:
                                    print(f"\r[识别中...] {utterance['text']}", end='', flush=True)
                                else:
                                    print(f"\n[最终结果] {utterance['text']}")
                                    update_recognition_text(utterance['text'])

# 添加其他必要的函数
def generate_full_default_header():
    return generate_header()

def generate_header(
    version=PROTOCOL_VERSION,
    message_type=CLIENT_FULL_REQUEST,
    message_type_specific_flags=NO_SEQUENCE,
    serial_method=JSON,
    compression_type=GZIP,
    reserved_data=0x00,
    extension_header=bytes()
):
    """生成请求头"""
    header = bytearray()
    header_size = int(len(extension_header) / 4) + 1
    header.append((version << 4) | header_size)
    header.append((message_type << 4) | message_type_specific_flags)
    header.append((serial_method << 4) | compression_type)
    header.append(reserved_data)
    header.extend(extension_header)
    return header

def generate_audio_default_header():
    """生成音频数据请求头"""
    return generate_header(message_type=CLIENT_AUDIO_ONLY_REQUEST)

def parse_response(res):
    """解析响应"""
    try:
        protocol_version = res[0] >> 4
        header_size = res[0] & 0x0f
        message_type = res[1] >> 4
        message_type_specific_flags = res[1] & 0x0f
        serialization_method = res[2] >> 4
        message_compression = res[2] & 0x0f
        reserved = res[3]
        header_extensions = res[4:header_size * 4]
        payload = res[header_size * 4:]
        result = {}
        payload_msg = None
        payload_size = 0
        
        if message_type == SERVER_FULL_RESPONSE:
            payload_size = int.from_bytes(payload[:4], "big", signed=True)
            payload_msg = payload[4:]
        elif message_type == SERVER_ACK:
            seq = int.from_bytes(payload[:4], "big", signed=True)
            result['seq'] = seq
            if len(payload) >= 8:
                payload_size = int.from_bytes(payload[4:8], "big", signed=False)
                payload_msg = payload[8:]
        elif message_type == SERVER_ERROR_RESPONSE:
            code = int.from_bytes(payload[:4], "big", signed=False)
            result['code'] = code
            payload_size = int.from_bytes(payload[4:8], "big", signed=False)
            payload_msg = payload[8:]
            
        if payload_msg is None:
            return result
            
        if message_compression == GZIP:
            payload_msg = gzip.decompress(payload_msg)
            
        if serialization_method == JSON:
            payload_msg = json.loads(str(payload_msg, "utf-8"))
        elif serialization_method != NO_SERIALIZATION:
            payload_msg = str(payload_msg, "utf-8")
            
        result['payload_msg'] = payload_msg
        result['payload_size'] = payload_size
        return result
    except Exception as e:
        return {"error": f"Failed to parse response: {str(e)}"}

def update_recognition_text(text, is_correction=False):
    """更新识别结果文本框"""
    try:
        if text_box and hasattr(text_box, 'text'):
            # 直接追加文本
            text_box.text.insert(END, f"{text}\n")
            text_box.text.see(END)
    except Exception as e:
        print(f"更新文本错误: {e}")

@app.errorhandler(404)
def not_found(error):
    return jsonify({"status": "error", "message": "接口不存在"}), 404

@app.errorhandler(500)
def server_error(error):
    return jsonify({"status": "error", "message": "服务器内部错误"}), 500

def cleanup():
    """程序退出时的清理工作"""
    global is_recording, loop
    try:
        is_recording = False
        if loop and not loop.is_closed():
            loop.close()
    except Exception as e:
        print(f"清理时出错: {e}")

def on_retry_name():
    """重试姓名识别按钮点击处理"""
    global recognized_name, is_recording
    try:
        print("重新开始姓名识别")
        
        # 清除已识别的姓名
        recognized_name = None
        
        # 清空文本框
        if hasattr(text_box, 'text'):
            text_box.text.delete("1.0", END)
            text_box.text.insert(END, "准备开始录音...\n")
            
        # 重置录音状态
        is_recording = False
        
        # 启动新的录音
        start_button_click()
        
    except Exception as e:
        print(f"重试姓名识别时出错: {e}")

def on_confirm_name():
    """确认姓名按钮点击处理"""
    global recognized_name
    try:
        if not recognized_name:
            print("没有可确认的姓名")
            update_recognition_text("请先点击姓名按钮进行识别")
            return
            
        print(f"姓名已确认: {recognized_name}")
        update_recognition_text(f"姓名 {recognized_name} 已确认\n可以继续录音...")
        
        # 这里可以添加其他确认后的操作
        # 比如保存到数据库等
        
    except Exception as e:
        print(f"确认姓名时出错: {e}")

if __name__ == '__main__':
    try:
        print("\n=== 程序启动顺序 ===")
        print("0. 初始化API密钥")
        print("1. 启动Web服务器")
        print("2. 初始化GUI界面")
        print("3. 启动语音识别\n")
        
        # 初始化API密钥
        if not init_api_keys():
            print("API密钥初始化失败，程序退出")
            sys.exit(1)
            
        # 启动Web服务器
        print("正在启动Web服务器...\n")
        flask_thread = threading.Thread(target=run_flask)
        flask_thread.daemon = True
        flask_thread.start()
        
        # 等待服务器启动
        time.sleep(2)
        
        # 初始化GUI
        print("\n正在初始化GUI...\n")
        gui, text_box = init_main_gui()
        
        # 主循环
        while True:
            try:
                if loop and not loop.is_closed():
                    loop.run_until_complete(asyncio.sleep(0.1))
                gui.update()
            except KeyboardInterrupt:
                print("\n程序被用户中断")
                break
            except Exception as e:
                print(f"\n循环中出现错误: {e}")
                break
                
    except Exception as e:
        print(f"\n程序异常: {e}")
    finally:
        cleanup()

# 设置API密钥
os.environ['DASHSCOPE_API_KEY'] = 'sk-7c04ee6f9432492bb344baa7a5c0162f'