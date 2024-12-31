import asyncio
from concurrent.futures import ThreadPoolExecutor
import json
import requests
from typing import List, Dict, Optional, Tuple

class CodeGenerationCoordinator:
    def __init__(self, zhipu_client, wenxin_client):
        """
        初始化代码生成协调器

        Args:
            zhipu_client: 智谱AI的客户端
            wenxin_client: 文心一言的客户端
        """
        self.zhipu = zhipu_client
        self.wenxin = wenxin_client
        self.executor = ThreadPoolExecutor(max_workers=2)
        self.iteration_history = []

    async def generate_code(self,
                            task_description: str,
                            language: str,
                            iterations: int = 3,
                            temperature: float = 0.7) -> Tuple[str, List[str]]:
        """
        通过多轮迭代生成和优化代码

        Args:
            task_description: 代码生成任务的描述
            language: 目标编程语言
            iterations: 迭代优化的次数
            temperature: 生成温度参数

        Returns:
            Tuple[str, List[str]]: (最终代码, 迭代历史)
        """
        self.iteration_history = []

        # 第一轮：获取思维链和初始代码
        cot = await self._get_cot(task_description)
        initial_code = await self._generate_initial_code(task_description, cot, language, temperature)
        self.iteration_history.append(("Initial Code", initial_code))

        current_code = initial_code

        # 多轮迭代优化
        for i in range(iterations):
            # 交替使用两个LLM进行优化
            optimized_code = await self._iterate_optimization(
                current_code,
                task_description,
                language,
                iteration=i,
                temperature=max(0.3, temperature - 0.1 * i)  # 随迭代降低温度
            )

            if optimized_code and self._is_better_version(current_code, optimized_code):
                current_code = optimized_code
                self.iteration_history.append((f"Iteration {i+1}", current_code))
            else:
                print(f"Iteration {i+1} did not produce better results, keeping previous version")

        # 最后一轮：格式化和清理
        final_code = await self._final_cleanup(current_code, language)
        if final_code:
            self.iteration_history.append(("Final Cleanup", final_code))
            return final_code, self.iteration_history

        return current_code, self.iteration_history

    async def _get_cot(self, task_description: str) -> str:
        """获取思维链(Chain of Thought)"""
        prompt = f"基于以下的目的给我一个使用Python解决问题的思维链(CoT)只需要告诉我第一步、第二步...分别需要做什么即可，不要输出多余信息: {task_description}"

        response = await self._call_llm(self.wenxin, prompt, temperature=0.7)
        return response

    async def _generate_initial_code(self,
                                     task_description: str,
                                     cot: str,
                                     language: str,
                                     temperature: float) -> str:
        """生成初始代码"""
        detailed_prompt = self._build_detailed_prompt(task_description, cot, language)

        # 并行调用两个LLM
        try:
            responses = await asyncio.gather(
                self._call_llm(self.wenxin, detailed_prompt, temperature),
                self._call_llm(self.llm2, detailed_prompt, temperature)
            )
            return self._analyze_and_merge_responses(responses)
        except Exception as e:
            print(f"Error during initial code generation: {e}")
            return None

    async def _iterate_optimization(self,
                                    code: str,
                                    task_description: str,
                                    language: str,
                                    iteration: int,
                                    temperature: float) -> str:
        """单轮迭代优化"""
        # 根据迭代轮次选择不同的优化重点
        optimization_focus = self._get_optimization_focus(iteration)

        prompt = self._build_iteration_prompt(
            code,
            task_description,
            language,
            optimization_focus,
            iteration
        )

        # 交替使用两个LLM
        llm_to_use = self.wenxin if iteration % 2 == 0 else self.llm2

        try:
            response = await self._call_llm(llm_to_use, prompt, temperature)
            return self._extract_code_block(response)
        except Exception as e:
            print(f"Error during iteration {iteration}: {e}")
            return None

    def _get_optimization_focus(self, iteration: int) -> List[str]:
        """根据迭代轮次确定优化重点"""
        focuses = [
            ["代码结构优化", "命名规范", "基础功能完善"],
            ["性能优化", "边界条件处理", "异常处理"],
            ["代码简化", "注释完善", "代码风格统一"],
        ]
        return focuses[iteration % len(focuses)]

    def _build_iteration_prompt(self,
                                code: str,
                                task_description: str,
                                language: str,
                                optimization_focus: List[str],
                                iteration: int) -> str:
        """构建迭代优化提示"""
        return f"""请基于以下内容对代码进行第{iteration + 1}轮优化：

任务描述：
{task_description}

当前代码：
{code}

本轮优化重点：
{', '.join(optimization_focus)}

请注意：
1. 保持代码的基本功能不变
2. 重点关注上述优化方向
3. 确保代码可以正常运行
4. 维护代码的可读性
5. 保留或优化现有的注释

请直接返回优化后的完整代码。
"""

    async def _final_cleanup(self, code: str, language: str) -> str:
        """最终的代码清理和格式化"""
        cleanup_prompt = f"""请对以下{language}代码进行最后的清理和格式化：

{code}

要求：
1. 统一代码风格
2. 确保命名规范
3. 优化代码格式
4. 完善注释
5. 不要改变代码功能

请返回清理后的代码。
"""
        try:
            response = await self._call_llm(self.wenxin, cleanup_prompt, temperature=0.3)
            return self._extract_code_block(response)
        except Exception as e:
            print(f"Error during final cleanup: {e}")
            return code

    def _is_better_version(self, old_code: str, new_code: str) -> bool:
        """评估新版本是否比旧版本更好"""
        if not new_code:
            return False

        # 检查代码长度变化不能过大
        if len(new_code) < len(old_code) * 0.5 or len(new_code) > len(old_code) * 2:
            return False

        # 确保基本语法正确
        try:
            compile(new_code, '<string>', 'exec')
        except SyntaxError:
            return False

        # 检查是否包含必要的组成部分（函数定义、类等）
        old_components = self._extract_code_components(old_code)
        new_components = self._extract_code_components(new_code)
        if len(new_components) < len(old_components) * 0.8:
            return False

        return True

    def _extract_code_components(self, code: str) -> List[str]:
        """提取代码中的主要组成部分"""
        components = []
        lines = code.split('\n')
        for line in lines:
            line = line.strip()
            if line.startswith('def ') or line.startswith('class '):
                components.append(line)
        return components

    async def _call_llm(self,
                        llm_client,
                        prompt: str,
                        temperature: float) -> str:
        """异步调用LLM API"""
        loop = asyncio.get_running_loop()
        return await loop.run_in_executor(
            self.executor,
            lambda: llm_client.chat.completions.create(
                model="glm-4",
                messages=[{
                    "role": "user",
                    "content": prompt
                }],
                temperature=temperature
            ).choices[0].message.content
        )
    def _analyze_and_merge_responses(self, responses: List[str]) -> str:
        """分析和合并两个LLM的响应"""
        codes = [self._extract_code_block(response) for response in responses]

        if not codes[0]:
            return codes[1]
        if not codes[1]:
            return codes[0]

        if len(codes[0]) > len(codes[1]) * 1.5:
            return codes[0]
        if len(codes[1]) > len(codes[0]) * 1.5:
            return codes[1]

        return codes[0]

    def _extract_code_block(self, text: str) -> str:
        """从响应中提取代码块"""
        text = text.strip()
        if text.startswith('```python'):
            text = text[8:]
        if text.startswith('```'):
            text = text[3:]
        if text.endswith('```'):
            text = text[:-3]
        return text.strip()