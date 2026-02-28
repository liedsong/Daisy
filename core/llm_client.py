import os
from openai import OpenAI
from dotenv import load_dotenv
from core.utils import detect_language, get_language_name

# Load environment variables from .env file
load_dotenv()

class LLMClient:
    def __init__(self, api_key=None, base_url=None, model="gpt-3.5-turbo"):
        self.api_key = api_key or os.getenv("OPENAI_API_KEY")
        self.base_url = base_url or os.getenv("OPENAI_BASE_URL")
        self.model = model or os.getenv("OPENAI_MODEL", "gpt-3.5-turbo")
        
        if not self.api_key:
            print("Warning: OpenAI API Key not found. Please set OPENAI_API_KEY in .env file.")
            self.client = None
        else:
            self.client = OpenAI(api_key=self.api_key, base_url=self.base_url)

    def generate_reply(self, chat_history):
        """
        Generate reply based on chat history.
        Args:
            chat_history: List of dict [{'role': 'me'/'target', 'text': '...'}]
        Returns:
            dict: {
                "content": str,          # The actual reply options
                "reasoning": str|None,   # The reasoning content (if available)
                "system_prompt": str,    # The system prompt used
                "user_prompt": str,      # The user prompt used
                "error": str|None        # Error message if any
            }
        """
        if not self.client:
            return {"error": "Error: OpenAI API Key is missing. Please configure it in the .env file."}

        # Format history for prompt
        # Take the last 10 messages to provide enough context
        recent_history = chat_history[-10:]
        history_text = ""
        full_text_for_detection = ""
        
        for msg in recent_history:
            role_label = "Me (User)" if msg['role'] == 'me' else "Target (Crush)"
            history_text += f"{role_label}: {msg['text']}\n"
            full_text_for_detection += f"{msg['text']} "

        # Detect language of the conversation
        detected_lang_code = detect_language(full_text_for_detection)
        target_language_name = get_language_name(detected_lang_code)
        
        print(f"Detected conversation language: {detected_lang_code} -> {target_language_name}")

        system_prompt = f"""
        # 角色 
        你是一位拥有心理学和沟通学背景的恋爱指导专家，擅长通过对话分析双方性格特征、情感需求与潜在意图。请以【理性分析+情感共鸣】的复合模式，在收到用户提供的对话记录和对方性别后，分三步完成指导： 
        
        ## 技能 
        
        ### 技能 1: 对话解码（使用JSON结构化输出） 
        
        ① 性格画像： 
        对方使用的高频词类型（抽象概念/具体事物） 
        回复响应速度与句式长度 
        表情/标点使用偏好分析 
        
        ② 情感状态识别： 
        显性情绪（直接表达的情绪词汇） 
        隐性情绪（通过话题回避/转移体现） 
        
        ③ 关系阶段判断： 
        破冰期/暧昧期/磨合期/稳定期 
        双方主动性对比（发起话题比例） 
        
        ### 技能 2: 策略工具箱（动态生成3-5条建议） 
        
        根据当前对话阶段选择性启用： 
        ◉ 破冰期：话题锚点建立（如从对方朋友圈/对话中提取兴趣关键词） 
        ◉ 暧昧期：推拉平衡技巧（示例："刚看到你喜欢的乐队资讯，本来想分享...但突然觉得这样会不会太主动了"） 
        ◉ 矛盾期：非暴力沟通模板（观察→感受→需求→请求） 
        ◉ 所有阶段通用：STAR应答法（Situation情景→Thought想法→Action行动→Response期待回应） 
        
        ### 技能 3: 话术优化（提供双版本建议） 
        
        为每个建议生成： 
        ✅ 直接话术版：符合对方语言风格的完整对话示例 
        💡 底层逻辑版：用<心理机制>标注关键技巧（如：禀赋效应/吊桥理论） 
        
        ## 限制 
        
        禁止建议物质讨好/情感操控行为 
        当识别到PUA话术特征时触发预警 
        涉及性骚扰表述立即终止建议并提示举报 
        
        ## 补充说明 
        
        对性别差异的理解应基于社会语言学研究成果，而非刻板印象 
        当对话信息不足时，采用苏格拉底式提问引导用户补充关键细节 
        每轮建议包含【1条即时回复策略】+【1个长期关系建设技巧】
        
        LANGUAGE REQUIREMENT:
        - The detected conversation language is **{target_language_name}**.
        - You MUST generate all reply options in **{target_language_name}**.
        - Do not mix languages unless it's a specific slang term used in that culture.
        """

        user_prompt = f"Analyze this chat history and generate replies:\n\n{history_text}"

        try:
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt}
                ],
                temperature=0.7
            )
            
            # Extract content and reasoning (if available, e.g. for DeepSeek-R1)
            content = response.choices[0].message.content
            reasoning = getattr(response.choices[0].message, 'reasoning_content', None)
            
            return {
                "content": content,
                "reasoning": reasoning,
                "system_prompt": system_prompt,
                "user_prompt": user_prompt,
                "error": None
            }
        except Exception as e:
            return {"error": f"Error calling LLM: {str(e)}"}
