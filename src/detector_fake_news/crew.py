"""Crew definition for the fake news detector."""

from __future__ import annotations

from typing import List

from crewai import Agent, Crew, Process, Task
from crewai.agents.agent_builder.base_agent import BaseAgent
from crewai.project import CrewBase, agent, crew, task

from detector_fake_news.llm import get_llm
from detector_fake_news.models import BiasReport, ClaimExtraction, FinalVerdict, LegalCase
from detector_fake_news.runtime import configure_runtime_environment
from detector_fake_news.tools import ArticleResearchTool

configure_runtime_environment()


@CrewBase
class FakeNewsCrew:
    """Sequential fake-news detection crew."""

    agents: List[BaseAgent]
    tasks: List[Task]

    agents_config = "config/agents.yaml"
    tasks_config = "config/tasks.yaml"

    @agent
    def claim_extractor(self) -> Agent:
        return Agent(
            config=self.agents_config["claim_extractor"],  # type: ignore[index]
            llm=get_llm(),
            verbose=True,
            allow_delegation=False,
            inject_date=True,
        )

    @agent
    def supporting_counsel(self) -> Agent:
        return Agent(
            config=self.agents_config["supporting_counsel"],  # type: ignore[index]
            llm=get_llm(),
            tools=[ArticleResearchTool()],
            verbose=True,
            allow_delegation=False,
            inject_date=True,
        )

    @agent
    def opposing_counsel(self) -> Agent:
        return Agent(
            config=self.agents_config["opposing_counsel"],  # type: ignore[index]
            llm=get_llm(),
            tools=[ArticleResearchTool()],
            verbose=True,
            allow_delegation=False,
            inject_date=True,
        )

    @agent
    def bias_analyst(self) -> Agent:
        return Agent(
            config=self.agents_config["bias_analyst"],  # type: ignore[index]
            llm=get_llm(),
            verbose=True,
            allow_delegation=False,
            inject_date=True,
        )

    @agent
    def judge(self) -> Agent:
        return Agent(
            config=self.agents_config["judge"],  # type: ignore[index]
            llm=get_llm(),
            verbose=True,
            allow_delegation=False,
            inject_date=True,
        )

    @task
    def extract_claims_task(self) -> Task:
        return Task(
            config=self.tasks_config["extract_claims_task"],  # type: ignore[index]
            agent=self.claim_extractor(),
            output_pydantic=ClaimExtraction,
        )

    @task
    def build_supporting_case_task(self) -> Task:
        return Task(
            config=self.tasks_config["build_supporting_case_task"],  # type: ignore[index]
            agent=self.supporting_counsel(),
            context=[self.extract_claims_task()],
            output_pydantic=LegalCase,
        )

    @task
    def build_opposing_case_task(self) -> Task:
        return Task(
            config=self.tasks_config["build_opposing_case_task"],  # type: ignore[index]
            agent=self.opposing_counsel(),
            context=[self.extract_claims_task()],
            output_pydantic=LegalCase,
        )

    @task
    def bias_analysis_task(self) -> Task:
        return Task(
            config=self.tasks_config["bias_analysis_task"],  # type: ignore[index]
            agent=self.bias_analyst(),
            output_pydantic=BiasReport,
        )

    @task
    def issue_verdict_task(self) -> Task:
        return Task(
            config=self.tasks_config["issue_verdict_task"],  # type: ignore[index]
            agent=self.judge(),
            context=[
                self.extract_claims_task(),
                self.build_supporting_case_task(),
                self.build_opposing_case_task(),
                self.bias_analysis_task(),
            ],
            output_pydantic=FinalVerdict,
        )

    @crew
    def crew(self) -> Crew:
        return Crew(
            agents=self.agents,
            tasks=self.tasks,
            process=Process.sequential,
            verbose=True,
        )


def build_crew() -> Crew:
    """Convenience helper for callers outside the CrewBase pattern."""
    return FakeNewsCrew().crew()
