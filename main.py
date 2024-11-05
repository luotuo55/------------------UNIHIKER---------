from flask import Flask, request, send_file, jsonify, Response
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
import queue
import random
from datetime import datetime
import re
from http import HTTPStatus

# Flask 应用初始化
app = Flask(__name__, 
    static_folder='.', # 设置当前目录为静态文件目录
    static_url_path='' # 将静态文件URL路径设为空
)

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
    """首页路由"""
    try:
        print("尝试访问首页")
        return send_file('index.html')  # 直接使用文件名，因为文件在同一目录
    except Exception as e:
        print(f"访问首页出错: {e}")
        return f"Error: {str(e)}", 500

@app.route('/submit', methods=['POST'])
def submit():
    """处理范文提交"""
    try:
        data = request.get_json()
        print(f"收到范文提交: {data}")
        
        if not data or 'text' not in data:
            return jsonify({
                "status": "error", 
                "message": "无效的范文数据"
            }), 400
            
        text = data['text'].strip()
        if not text:
            return jsonify({
                "status": "error", 
                "message": "范文不能为空"
            }), 400
            
        # 保存范文
        shared_data.target_text = text
        print(f"范文内容已保存: {text}")
        
        # 验证是否保存成功
        if shared_data.target_text == text:
            return jsonify({
                "status": "success", 
                "message": "范文已成功保存"
            })
        else:
            return jsonify({
                "status": "error", 
                "message": "范文保存失败"
            }), 500
            
    except Exception as e:
        print(f"处理范文出错: {e}")
        return jsonify({
            "status": "error", 
            "message": f"服务器错误: {str(e)}"
        }), 500

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
scores = []  # 存储所有分数记录
recognized_name = None  # 当前识别的学生姓名

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

def call_qwen_api(text, mode='correction'):
    """统一的千问API调用函数"""
    try:
        if not dashscope.api_key:
            print("[错误] DashScope API Key未设置")
            return None
            
        if mode == 'scoring':
            current_text = text.get('speech_text', '')
            target_text = text.get('target_text', '')
            messages = [
                {
                    'role': 'system',
                    'content': '你是一个专业的朗读评分助手。请只返回三个分数，格式固定。'
                },
                {
                    'role': 'user',
                    'content': f"""
请对比以下两段文本进行评分（满分100分）。只需返回三个分数，格式如下：
准确度：数字
完整度：数字
流畅度：数字

范文：{target_text}
朗读文本：{current_text}
"""
                }
            ]
        else:  # correction mode
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
            result_format='message'
        )
        
        if response.status_code == 200:
            result = response.output.choices[0].message.content.strip()
            print(f"[处理结果] {result}")
            return result
        else:
            print(f"[错误] API调用失败: {response.code} - {response.message}")
            return None
            
    except Exception as e:
        print(f"[错误] API调用失败: {str(e)}")
        return None

def on_correction_click():
    """处理纠正按钮点击事件"""
    global is_recording
    try:
        print("纠正按钮被点击")
        
        # 1. 停止语音识别
        is_recording = False
        print("正在停止语音识别...")
        time.sleep(1)
        
        # 2. 获取当前文本
        current_text = text_box.text.get("1.0", END).strip()
        print(f"获取到的当前文本：{current_text}")
        
        if current_text:
            # 3. 调用API进行纠正
            print("开始调用API进行纠正...")
            result = call_qwen_api(current_text, mode='correction')
            
            if result:
                # 4. 显示纠正结果
                update_recognition_text(f"[纠正结果]\n{result}")
                print("更新显示完成")
            else:
                update_recognition_text("[错误] 文本纠正失败")
        else:
            print("文本框为空，不进行API调用")
            update_recognition_text("[错误] 请先进行语音识别")
            
    except Exception as e:
        error_msg = f"纠正文本错误: {str(e)}"
        print(error_msg)
        update_recognition_text(error_msg)

def on_score_click():
    """处理打分按钮点击事件"""
    global is_recording, scores, recognized_name
    try:
        print("打分按钮被点击")
        
        # 1. 停止语音识别
        is_recording = False
        time.sleep(1)
        
        # 2. 获取文本
        current_text = text_box.text.get("1.0", END).strip()
        target_text = shared_data.target_text
        
        if not target_text or not current_text:
            error_msg = "[错误] 请确保已输入范文且已完成语音识别"
            print(error_msg)
            update_recognition_text(error_msg)
            return
            
        if not recognized_name:
            error_msg = "[错误] 请先识别姓名"
            print(error_msg)
            update_recognition_text(error_msg)
            return
            
        # 3. 调用API进行评分
        print("开始评分...")
        result = call_qwen_api({
            'speech_text': current_text,
            'target_text': target_text
        }, mode='scoring')
        
        if result:
            # 解析分数
            score_data = {}
            for line in result.split('\n'):
                if ':' in line:
                    key, value = line.split(':', 1)
                    key = key.strip()
                    value = int(''.join(filter(str.isdigit, value.strip())))
                    score_data[key] = value
            
            # 构建分数记录
            score_record = {
                'name': recognized_name,
                'accuracy': score_data.get('准确度', 0),
                'fluency': score_data.get('流畅度', 0),
                'completeness': score_data.get('完整度', 0),
                'timestamp': time.strftime("%Y-%m-%d %H:%M:%S")
            }
            
            # 添加到分数列表
            scores.append(score_record)
            
            # 更新显示
            display_text = (
                f"\n[评分结果]\n"
                f"姓名：{score_record['name']}\n"
                f"准确度：{score_record['accuracy']}\n"
                f"流畅度：{score_record['fluency']}\n"
                f"完整度：{score_record['completeness']}"
            )
            update_recognition_text(display_text)
            print("评分完成")
            
        else:
            update_recognition_text("[错误] 评分失败")
            
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
    """理姓名按钮点击事件"""
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
            print("文为空，无法识别姓")
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
        print("正在启动Flask服务器...")
        app.run(host='0.0.0.0', port=5000, debug=False)
    except Exception as e:
        print(f"Flask服务器启动错误: {e}")

def get_ip_address():
    """获取机IP地址"""
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
                print("初始化功")
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
    """生成频数据请求头"""
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

def update_recognition_text(new_text, is_correction=False):
    """更新识别结果显示"""
    try:
        print(f"\n准备更新显示文本: {new_text[:100]}...")  # 只打印前100个字符
        
        # 确保text_box存在
        if not hasattr(text_box, 'text'):
            print("错误：text_box.text不存在")
            return
            
        # 添加到文本列表
        all_texts.append(new_text)
        
        # 构建完整文本
        full_text = "\n".join(all_texts)
        
        # 更新显示
        text_box.config(text=full_text)
        text_box.text.see(END)  # 滚动到底部
        
        # 刷新GUI
        if gui and hasattr(gui, 'master') and gui.master.winfo_exists():
            gui.update()
            print("显示已更新")
        else:
            print("警告：GUI不存在或已关闭")
            
    except Exception as e:
        print(f"更新显示文本时出错: {str(e)}")

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
            text_box.text.insert(END, "准备开始录...\n")
            
        # 重置录音状态
        is_recording = False
        
        # 启动新的录音
        start_button_click()
        
    except Exception as e:
        print(f"重试姓识别时出错: {e}")

def on_confirm_name():
    """确认姓名按钮点击处理"""
    global recognized_name
    try:
        if not recognized_name:
            print("没有可确认的姓名")
            update_recognition_text("请先点击姓名按钮进行识别")
            return False
            
        print(f"姓名已确认: {recognized_name}")
        update_recognition_text(f"姓名 {recognized_name} 已确认\n可以继续录音...")
        return True
        
    except Exception as e:
        print(f"确认姓名时出错: {e}")
        return False

@app.route('/get_scores')
def get_scores():
    """获取所有分数记录"""
    global scores
    return jsonify(scores)

@app.route('/clear_scores', methods=['POST'])
def clear_scores():
    """清除所有分数记录"""
    global scores
    try:
        scores = []
        return jsonify({"status": "success", "message": "分数记录已清除"})
    except Exception as e:
        return jsonify({"status": "error", "message": str(e)}), 500

# 添加CORS支持
@app.after_request
def after_request(response):
    response.headers.add('Access-Control-Allow-Origin', '*')
    response.headers.add('Access-Control-Allow-Headers', 'Content-Type')
    response.headers.add('Access-Control-Allow-Methods', 'GET,POST')
    return response

@app.route('/score', methods=['POST'])
def handle_score():
    """处理打分请求"""
    try:
        if not recognized_name:
            return jsonify({"status": "error", "message": "请先识别姓名"}), 400
            
        if not corrected_text or not reference_text:
            return jsonify({"status": "error", "message": "请先完成朗读"}), 400
            
        # 获取千文评分
        score_results = get_scores_from_qianwen(corrected_text, reference_text)
        if not score_results:
            return jsonify({"status": "error", "message": "评分失败"}), 500
            
        # 生成分数记录
        score_data = {
            'name': recognized_name,
            'accuracy': score_results['accuracy'],
            'fluency': score_results['fluency'],
            'completeness': score_results['completeness'],
            'timestamp': time.strftime("%Y-%m-%d %H:%M:%S")
        }
        
        # 添加到scores列表
        scores.append(score_data)
        
        # 显示在UNIHIKER屏幕上
        score_text = (
            f"\n【评分结果】\n"
            f"姓名：{score_data['name']}\n"
            f"准确度：{score_data['accuracy']}\n"
            f"流畅度：{score_data['fluency']}\n"
            f"完整度：{score_data['completeness']}"
        )
        update_recognition_text(score_text)
        
        return jsonify({"status": "success", "data": score_data})
        
    except Exception as e:
        return jsonify({"status": "error", "message": str(e)}), 500

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