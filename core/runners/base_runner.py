#!/usr/bin/env python3

import asyncio
import logging
import time
from abc import ABC, abstractmethod
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Any, Optional

from ..interfaces import (
    IExperimentRunner, IAgent, ITopology, IConsensusEngine, 
    IEvaluator, IVisualizer, IDataLoader, IResultProcessor,
    MethodType, AgentType, TopologyType, ConsensusMethod,
    QuestionData, AgentResponse, ConsensusResult, ExperimentResult
)
from ..agents.agent_factory import get_agent_factory
from ..consensus.consensus_engine import create_consensus_engine
from ..agents.base_agent import Message
from ..data.data_loader import DataLoaderFactory
from ..results.result_processor import create_result_processor
from ..evaluation.unified_metrics import UnifiedByzantineMetrics
from ..topologies.topology_factory import TopologyFactory
from ..experiment_manager.position_controller import MaliciousNodePositionController

logger = logging.getLogger(__name__)

REPO_ROOT = Path(__file__).resolve().parents[2]

class BaseMethodRunner(IExperimentRunner):

    def __init__(self, config: Any):
        self.config = config
        self.method_type = self._get_method_type()

        self.agents: Dict[str, IAgent] = {}
        self.topology: Optional[ITopology] = None
        self.consensus_engine: Optional[IConsensusEngine] = None
        self.evaluator: Optional[IEvaluator] = None
        self.visualizer: Optional[IVisualizer] = None
        self.data_loader: Optional[IDataLoader] = None
        self.result_processor: Optional[IResultProcessor] = None

        self.questions: List[QuestionData] = []
        self.experiment_result: Optional[ExperimentResult] = None

        self.is_setup = False
        self.start_time = 0.0

        logger.info(f"初始化方法运行器: {self.method_type.value}")

    @property
    def method_type(self) -> MethodType:

        return self._method_type

    @method_type.setter
    def method_type(self, value: MethodType):

        self._method_type = value

    @abstractmethod
    def _get_method_type(self) -> MethodType:

        pass

    async def setup_experiment(self, config: Any = None) -> None:

        if config:
            self.config = config

        logger.info("开始设置实验环境...")

        try:

            await self._initialize_components()

            await self._load_data()

            await self._create_agents()

            await self._setup_topology()

            await self._setup_malicious_agents()

            self._validate_setup()

            self.is_setup = True
            logger.info("实验环境设置完成")

        except Exception as e:
            logger.error(f"实验环境设置失败: {e}")
            raise

    async def run_experiment(self) -> ExperimentResult:

        if not self.is_setup:
            await self.setup_experiment()

        self.start_time = time.time()
        logger.info(f"开始运行{self.method_type.value}实验...")

        try:

            consensus_results = await self._run_experiment_core()

            evaluation_metrics = await self._evaluate_results(consensus_results)

            execution_time = time.time() - self.start_time
            self.experiment_result = self._create_experiment_result(
                consensus_results, evaluation_metrics, execution_time
            )

            if self.config.save:
                await self._save_results()

            if self.config.visualize:
                await self._generate_visualizations()

            logger.info(f"实验完成，耗时: {execution_time:.2f}秒")
            return self.experiment_result

        except Exception as e:
            logger.error(f"实验运行失败: {e}")
            raise

    def cleanup_experiment(self) -> None:

        logger.info("清理实验环境...")

        for agent in self.agents.values():
            agent.reset()

        self.is_setup = False
        self.start_time = 0.0

        logger.info("实验环境清理完成")

    async def _initialize_components(self) -> None:

        logger.debug("初始化核心组件...")

        self.data_loader = DataLoaderFactory.create_data_loader(self.method_type)

        if (getattr(self.config, 'agent_type', '') == 'traditional' and not hasattr(self.config, 'consensus_method')):
            consensus_method = 'majority'
        elif (self.method_type == MethodType.PILOT and getattr(self.config, 'agent_type', '') == 'llm'):
            consensus_method = 'majority'
        elif (self.method_type == MethodType.DECODER):

            consensus_method = 'confidence_weighted'
        else:
            consensus_method = getattr(self.config, 'consensus_method', 'confidence_weighted')
        self.consensus_engine = create_consensus_engine(
            ConsensusMethod(consensus_method),
            convergence_threshold=getattr(self.config, 'convergence_threshold', 0.0),
            max_rounds=getattr(self.config, 'max_consensus_rounds', 5)
        )

        self.evaluator = UnifiedByzantineMetrics()

        from ..utils.naming import build_results_dir

        base_root = str(REPO_ROOT / "results")
        if self.method_type == MethodType.PILOT:

            method_name = 'pilot'
            dataset = str(getattr(self.config, 'dataset_type', 'gsm8k')).lower()
            agent_type = str(getattr(self.config, 'agent_type', 'llm')).lower()
            topo = str(getattr(self.config, 'topology', 'unknown'))
            output_dir = build_results_dir(
                base_root, method_name, dataset, agent_type, topo,
                int(self.config.agents), int(self.config.malicious)
            )
        elif self.method_type in (MethodType.PROMPT_PROBE, MethodType.DECODER):

            if self.method_type == MethodType.PROMPT_PROBE:
                method_name = 'prompt'
            else:
                method_name = 'decoder'
            dataset = str(getattr(self.config, 'dataset_type', 'gsm8k')).lower()
            agent_type = str(getattr(self.config, 'agent_type', 'llm')).lower()
            topo = str(getattr(self.config, 'topology', 'unknown'))
            output_dir = build_results_dir(
                base_root, method_name, dataset, agent_type, topo,
                int(self.config.agents), int(self.config.malicious)
            )
        else:

            output_dir = getattr(self.config, 'output_dir', base_root)
        try:

            setattr(self.config, 'output_dir', output_dir)
        except Exception as _e:
            logger.debug(f"结果保存时同步配置 output_dir 失败: {_e}")
        self.result_processor = create_result_processor(output_dir)

        self.visualizer = await self._create_visualizer()

        logger.debug("核心组件初始化完成")

    async def _load_data(self) -> None:

        logger.debug("加载实验数据...")

        data_path = self._get_data_path()
        data_kwargs = self._get_data_kwargs()

        self.questions = self.data_loader.load_questions(data_path, **data_kwargs)

        if not self.questions:
            raise ValueError("未能加载任何问题数据")

        if not self.data_loader.validate_data(self.questions):
            raise ValueError("数据验证失败")

        if self.config.mode == "single" and self.config.question:
            self.questions = [q for q in self.questions if q.question_id == self.config.question]
            if not self.questions:
                raise ValueError(f"未找到指定问题: {self.config.question}")

        logger.info(f"成功加载 {len(self.questions)} 个问题")

    async def _create_agents(self) -> None:

        logger.debug("创建智能体...")

        agent_factory = get_agent_factory()
        agent_type = AgentType(self.config.agent_type)

        try:
            if self.method_type == MethodType.DECODER:
                agent_type = AgentType.DECODER
        except Exception:
            pass

        for i in range(self.config.agents):
            agent_id = f"agent_{i}"

            extra_kwargs = {}

            try:
                if agent_type == AgentType.LLM and self.method_type == MethodType.PILOT:

                    extra_kwargs["role"] = "strong"
            except Exception:
                pass

            agent = agent_factory.create_agent(
                agent_id=agent_id,
                agent_type=agent_type,
                method_type=self.method_type,
                config=self.config,
                **extra_kwargs
            )

            self.agents[agent_id] = agent

        logger.info(f"成功创建 {len(self.agents)} 个智能体")

    async def _setup_topology(self) -> None:

        logger.debug("设置网络拓扑...")

        topology_factory = TopologyFactory()
        topology_type = TopologyType(self.config.topology)

        self.topology = topology_factory.create_topology(
            topology_type=topology_type,
            node_count=self.config.agents,
            seed=getattr(self.config, 'seed', None)
        )

        agent_ids = list(self.agents.keys())
        connections = self.topology.get_all_connections()

        for i, agent_id in enumerate(agent_ids):
            node_id = f"agent_{i}"
            if node_id in connections:
                neighbor_indices = [int(nid.split('_')[1]) for nid in connections[node_id]]
                neighbor_ids = [agent_ids[idx] for idx in neighbor_indices if idx < len(agent_ids)]
                self.agents[agent_id].set_neighbors(neighbor_ids)

        logger.info(f"拓扑设置完成: {topology_type.value}")

    async def _setup_malicious_agents(self) -> None:

        if self.config.malicious <= 0:
            logger.info("无恶意智能体")
            return

        logger.debug("设置恶意智能体...")

        position_controller = MaliciousNodePositionController()

        if self.config.specific_positions:
            malicious_positions = self.config.specific_positions
        else:
            malicious_positions = position_controller.select_malicious_nodes(
                topology_type=self.topology.topology_type,
                topology_instance=self.topology,
                num_malicious=self.config.malicious,
                position_strategy=self.config.position_strategy,
                seed=getattr(self.config, 'position_seed', None)
            )

        agent_ids = list(self.agents.keys())
        malicious_agents = []

        for pos in malicious_positions:
            if pos < len(agent_ids):
                agent_id = agent_ids[pos]
                self.agents[agent_id].set_malicious(True)
                malicious_agents.append(agent_id)

        logger.info(f"设置恶意智能体: {malicious_agents}")

    def _validate_setup(self) -> None:

        if not self.agents:
            raise ValueError("未创建任何智能体")

        if not self.questions:
            raise ValueError("未加载任何问题")

        if not self.topology:
            raise ValueError("未设置拓扑")

        if not self.consensus_engine:
            raise ValueError("未初始化共识引擎")

        logger.debug("设置验证通过")

    async def _run_experiment_core(self) -> List[ConsensusResult]:

        logger.info(f"开始处理 {len(self.questions)} 个问题...")

        consensus_results = []

        for i, question in enumerate(self.questions):
            logger.info(f"处理问题 {i+1}/{len(self.questions)}: {question.question_id}")

            for round_num in range(self.config.rounds):
                logger.debug(f"第 {round_num+1} 轮共识")

                agent_responses = await self._collect_agent_responses(question)

                should_do_consensus = False
                if self.method_type in (MethodType.PILOT, MethodType.PROMPT_PROBE):
                    should_do_consensus = (self.config.agent_type == 'llm')
                elif self.method_type == MethodType.DECODER:
                    should_do_consensus = True                      

                if should_do_consensus:

                    agent_ids = list(self.agents.keys())
                    initial_answer_map: Dict[str, str] = {}

                    is_decoder = (self.method_type == MethodType.DECODER)

                    response_by_agent: Dict[str, AgentResponse] = {r.agent_id: r for r in agent_responses}

                    if is_decoder:

                        neighbor_responses_per_agent: Dict[str, List[AgentResponse]] = {aid: [] for aid in agent_ids}
                        for aid in agent_ids:
                            resp = response_by_agent.get(aid)
                            if resp is None:
                                continue
                            initial_answer_map[aid] = str(getattr(resp, 'answer', ''))
                            neighbors = []
                            try:
                                neighbors = self.topology.get_neighbors(aid)
                            except Exception:
                                neighbors = []
                            for nid in neighbors:
                                if nid in neighbor_responses_per_agent:
                                    neighbor_responses_per_agent[nid].append(resp)
                    else:

                        neighbor_msgs_per_agent: Dict[str, List[Message]] = {aid: [] for aid in agent_ids}
                        for aid in agent_ids:
                            resp = response_by_agent.get(aid)
                            if resp is None:
                                continue
                            initial_answer_map[aid] = str(getattr(resp, 'answer', ''))
                            neighbors = []
                            try:
                                neighbors = self.topology.get_neighbors(aid)
                            except Exception:
                                neighbors = []
                            for nid in neighbors:
                                if nid in neighbor_msgs_per_agent:
                                    neighbor_msgs_per_agent[nid].append(
                                        Message(
                                            sender_id=aid,
                                            receiver_id=nid,
                                            message_type="answer",
                                            content={
                                                'answer': str(getattr(resp, 'answer', '')),
                                                'confidence': float(getattr(resp, 'confidence', 0.0) or 0.0)
                                            }
                                        )
                                    )

                    refined_responses: List[AgentResponse] = []
                    for aid in agent_ids:
                        agent = self.agents[aid]
                        final_resp = None
                        try:
                            if is_decoder:

                                final_resp = await agent.participate_in_consensus(
                                    question,
                                    neighbor_responses_per_agent.get(aid, [])
                                )
                            else:

                                final_resp = await agent.participate_in_consensus(
                                    question,
                                    neighbor_msgs_per_agent.get(aid, [])
                                )
                        except Exception as e:
                            logger.warning(f"智能体 {aid} 二次共识失败: {e}")
                            init_resp = next((r for r in agent_responses if r.agent_id == aid), None)
                            final_answer = getattr(init_resp, 'answer', '') if init_resp else ''
                            confidence = 0.0

                            refined_metadata = {
                                'initial_answer': initial_answer_map.get(aid, ''),
                                'final_answer': str(final_answer),
                                'error': str(e)
                            }
                            final_resp = AgentResponse(
                                agent_id=aid,
                                question_id=question.question_id,
                                answer=str(final_answer),
                                confidence=float(confidence),
                                reasoning=None,
                                metadata=refined_metadata,
                            )

                        if final_resp is not None:
                            refined_responses.append(final_resp)

                    agent_responses = refined_responses

                consensus_result = await self.consensus_engine.run_consensus(
                    question, agent_responses
                )

                consensus_results.append(consensus_result)

                if consensus_result.convergence_achieved:
                    logger.debug("共识收敛，提前结束")
                    break

        logger.info(f"完成 {len(consensus_results)} 个共识结果")
        return consensus_results

    async def _collect_agent_responses(self, question: QuestionData) -> List[AgentResponse]:

        responses = []

        tasks = []
        for agent in self.agents.values():
            task = agent.solve_problem(question)
            tasks.append(task)

        agent_responses = await asyncio.gather(*tasks, return_exceptions=True)

        for i, (agent_id, response) in enumerate(zip(self.agents.keys(), agent_responses)):
            if isinstance(response, Exception):
                logger.warning(f"智能体 {agent_id} 响应失败: {response}")

                response = AgentResponse(
                    agent_id=agent_id,
                    question_id=question.question_id,
                    answer="",
                    confidence=0.0,
                    reasoning="响应失败"
                )

            try:
                if getattr(response, 'metadata', None) is None:
                    response.metadata = {}

                if 'before_degree1_answer' not in response.metadata:
                    response.metadata['before_degree1_answer'] = str(response.answer)
                    response.metadata['degree1_switched'] = False
            except Exception:
                pass

            responses.append(response)

        try:
            if str(getattr(self.config, 'agent_type', '')).lower() == 'traditional':
                from random import random
                id_to_response = {r.agent_id: r for r in responses}
                trigger_count = 0
                for agent_id, agent in self.agents.items():

                    if getattr(agent, 'is_malicious', False):
                        continue

                    if not hasattr(self, 'topology') or self.topology is None:
                        continue
                    neighbors = []
                    try:
                        neighbors = self.topology.get_neighbors(agent_id)
                    except Exception:
                        neighbors = []
                    if len(neighbors) == 1:
                        own_resp = id_to_response.get(agent_id)
                        neigh_id = neighbors[0]
                        neigh_resp = id_to_response.get(neigh_id)

                        if own_resp and neigh_resp and str(own_resp.answer) != str(neigh_resp.answer) and random() < 0.5:
                            adjusted = AgentResponse(
                                agent_id=own_resp.agent_id,
                                question_id=own_resp.question_id,
                                answer=str(neigh_resp.answer),
                                confidence=0.0,
                                reasoning=getattr(own_resp, 'reasoning', ''),
                                response_time=getattr(own_resp, 'response_time', 0.0),
                                metadata=dict(getattr(own_resp, 'metadata', {}) or {})
                            )
                            try:
                                if adjusted.metadata is None:
                                    adjusted.metadata = {}
                                if 'before_degree1_answer' not in adjusted.metadata:
                                    adjusted.metadata['before_degree1_answer'] = str(own_resp.answer)
                                adjusted.metadata['degree1_switched'] = True
                                adjusted.metadata['degree1_neighbor'] = neigh_id
                            except Exception:
                                pass
                            id_to_response[agent_id] = adjusted
                            trigger_count += 1
                if trigger_count:
                    logger.info(f"度为1随机切换规则触发次数: {trigger_count}")
                responses = [id_to_response[r.agent_id] for r in responses]
        except Exception as e:
            logger.warning(f"度为1节点随机选择规则检查失败: {e}")

        return responses

    async def _evaluate_results(self, consensus_results: List[ConsensusResult]) -> Dict[str, Any]:

        logger.debug("评估实验结果...")

        agent_responses = {}
        correct_answers = []

        question_map = {q.question_id: q for q in self.questions}

        for cr in consensus_results:
            if cr.question_id in question_map:
                correct_answers.append(question_map[cr.question_id].correct_answer)

                for response in cr.individual_responses:
                    if response.agent_id not in agent_responses:
                        agent_responses[response.agent_id] = []
                    agent_responses[response.agent_id].append(response.answer)

        malicious_agents = [aid for aid, agent in self.agents.items() if agent.is_malicious]

        def _norm_safe(s: str) -> str:
            t = str(s).strip().lower()
            if t in ("1", "safe"):
                return "safe"
            if t in ("0", "unsafe"):
                return "unsafe"
            return str(s)

        def _map_commonsense_answer(answer: str, choices: list) -> str:

            if not choices:
                return answer
            answer_str = str(answer).strip()

            if len(answer_str) == 1 and answer_str.upper() in 'ABCDE':
                idx = ord(answer_str.upper()) - 65                 
                if 0 <= idx < len(choices):
                    return choices[idx]
            return answer

        try:
            dataset_type = getattr(self.config, 'dataset_type', 'gsm8k')
            if str(dataset_type).lower() == 'safe':
                correct_answers = [_norm_safe(a) for a in correct_answers]
                agent_responses = {aid: [_norm_safe(x) for x in xs] for aid, xs in agent_responses.items()}
            elif str(dataset_type).lower() == 'commonsense':

                logger.debug("CommonsenseQA数据集，映射字母答案为选项文本")

                for idx, cr in enumerate(consensus_results):
                    if cr.question_id in question_map:
                        question = question_map[cr.question_id]
                        choices = question.metadata.get('choices', []) if question.metadata else []

                        if choices:

                            for response in cr.individual_responses:
                                original_answer = response.answer
                                mapped_answer = _map_commonsense_answer(response.answer, choices)
                                if mapped_answer != original_answer:
                                    logger.debug(f"映射答案: {original_answer} → {mapped_answer}")
                                response.answer = mapped_answer

                            original_consensus = cr.consensus_answer
                            mapped_consensus = _map_commonsense_answer(cr.consensus_answer, choices)
                            if mapped_consensus != original_consensus:
                                logger.debug(f"映射共识答案: {original_consensus} → {mapped_consensus}")
                            cr.consensus_answer = mapped_consensus

                            correct_answer = question.correct_answer
                            cr.is_correct = (cr.consensus_answer.strip().lower() == correct_answer.strip().lower())
                            logger.debug(f"重新计算is_correct: {cr.consensus_answer} vs {correct_answer} = {cr.is_correct}")

                agent_responses = {}
                for cr in consensus_results:
                    for response in cr.individual_responses:
                        if response.agent_id not in agent_responses:
                            agent_responses[response.agent_id] = []
                        agent_responses[response.agent_id].append(response.answer)
        except Exception as _e:
            logger.warning(f"答案格式标准化失败: {_e}")

        consensus_answers: List[str] = []
        for cr in consensus_results:
            ans = str(cr.consensus_answer)
            consensus_answers.append(_norm_safe(ans) if str(getattr(self.config, 'dataset_type', 'gsm8k')).lower() == 'safe' else ans)

        evaluation_result = self.evaluator.generate_comprehensive_evaluation(
            agent_responses=agent_responses,
            correct_answers=correct_answers,
            malicious_agents=malicious_agents,
            use_academic_standards=False,
            consensus_results=consensus_answers
        )

        logger.debug("结果评估完成")
        return evaluation_result

    def _create_experiment_result(
        self, 
        consensus_results: List[ConsensusResult],
        evaluation_metrics: Dict[str, Any],
        execution_time: float
    ) -> ExperimentResult:

        if self.method_type == MethodType.PILOT:
            from ..utils.naming import generate_experiment_prefix
            method_name = 'pilot'
            dataset = str(getattr(self.config, 'dataset_type', 'gsm8k')).lower()
            agent_type = str(getattr(self.config, 'agent_type', 'llm')).lower()
            topo = str(getattr(self.config, 'topology', 'unknown'))
            experiment_id = generate_experiment_prefix(method_name, dataset, agent_type, topo,
                                                       int(self.config.agents), int(self.config.malicious))
        else:
            experiment_id = f"{self.method_type.value}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"

        return ExperimentResult(
            experiment_id=experiment_id,
            method_type=self.method_type,
            topology_type=TopologyType(self.config.topology),
            agent_count=self.config.agents,
            malicious_count=self.config.malicious,
            questions=self.questions,
            consensus_results=consensus_results,
            evaluation_metrics=evaluation_metrics,
            execution_time=execution_time,
            metadata={
                "config": self._serialize_config(),
                "malicious_agents": [aid for aid, agent in self.agents.items() if agent.is_malicious]
            }
        )

    async def _save_results(self) -> None:

        if not self.experiment_result:
            return

        logger.debug("保存实验结果...")

        base_output_dir = Path(getattr(self.config, 'output_dir', 'results'))
        exp_dir = base_output_dir / self.experiment_result.experiment_id
        exp_dir.mkdir(parents=True, exist_ok=True)
        output_file = exp_dir / f"{self.experiment_result.experiment_id}.json"

        self.result_processor.save_experiment_result(self.experiment_result, str(output_file))

        if getattr(self.config, 'export_csv', False):
            csv_file = output_file.with_suffix('.csv')
            self.result_processor.export_to_format(self.experiment_result, 'csv', str(csv_file))

        if getattr(self.config, 'export_excel', False):
            excel_file = output_file.with_suffix('.xlsx')
            self.result_processor.export_to_format(self.experiment_result, 'xlsx', str(excel_file))

        logger.info(f"结果已保存: {output_file}")

    async def _generate_visualizations(self) -> None:

        if not self.visualizer or not self.experiment_result:
            return

        logger.debug("生成可视化...")

        output_dir = Path(getattr(self.config, 'output_dir', 'results')) / self.experiment_result.experiment_id
        output_dir.mkdir(parents=True, exist_ok=True)

        self.visualizer.generate_comprehensive_report(self.experiment_result, str(output_dir))

        logger.info(f"可视化已生成: {output_dir}")

    @abstractmethod
    def _get_data_path(self) -> str:

        pass

    @abstractmethod
    def _get_data_kwargs(self) -> Dict[str, Any]:

        pass

    @abstractmethod
    async def _create_visualizer(self) -> Optional[IVisualizer]:

        pass

    def _serialize_config(self) -> Dict[str, Any]:

        config_dict = {}
        for attr in dir(self.config):
            if not attr.startswith('_'):
                value = getattr(self.config, attr)
                if isinstance(value, (str, int, float, bool, list, dict)):
                    config_dict[attr] = value
                elif hasattr(value, 'value'):        
                    config_dict[attr] = value.value
                else:
                    config_dict[attr] = str(value)
        return config_dict

if __name__ == "__main__":
    print("标准化方法运行器基类")
    print("此基类提供了统一的实验运行框架")
    print("所有具体方法都应继承此基类并实现抽象方法")
