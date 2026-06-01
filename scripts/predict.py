import os
import sys

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from mydatasets.base_dataset import BaseDataset
from agents.mdoc_agent import MDocAgent
import hydra


@hydra.main(config_path="../config", config_name="base", version_base="1.2")
def main(cfg):
    os.environ["CUDA_VISIBLE_DEVICES"] = cfg.mdoc_agent.cuda_visible_devices
    os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "max_split_size_mb:64"
    runtime_cfg = cfg.get("runtime", {})
    runtime_temperature = runtime_cfg.get("temperature", None)

    for agent_config in cfg.mdoc_agent.agents:
        agent_name = agent_config.agent
        model_name = agent_config.model

        agent_cfg = hydra.compose(
            config_name="agent/" + agent_name,
            overrides=[]
        ).agent

        model_cfg = hydra.compose(
            config_name="model/" + model_name,
            overrides=[]
        ).model
        if runtime_temperature is not None:
            model_cfg.temperature = runtime_temperature

        agent_config.agent = agent_cfg
        agent_config.model = model_cfg

    cfg.mdoc_agent.sum_agent.agent = hydra.compose(
        config_name="agent/" + cfg.mdoc_agent.sum_agent.agent,
        overrides=[]
    ).agent

    cfg.mdoc_agent.sum_agent.model = hydra.compose(
        config_name="model/" + cfg.mdoc_agent.sum_agent.model,
        overrides=[]
    ).model
    if runtime_temperature is not None:
        cfg.mdoc_agent.sum_agent.model.temperature = runtime_temperature

    dataset = BaseDataset(cfg.dataset)
    mdoc_agent = MDocAgent(cfg.mdoc_agent)

    # Optional resume path.
    # 用于从已有预测结果 JSON 继续补跑。
    # 如果没有传 runtime.resume_path，则默认为 None，行为和原来完全一致。
    resume_path = runtime_cfg.get("resume_path", None)

    mdoc_agent.predict_dataset(
        dataset,
        resume_path=resume_path
    )


if __name__ == "__main__":
    main()
