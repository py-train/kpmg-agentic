"""
CrewAI demo: 3 agents (Researcher, Writer, Editor) collaborating
in a sequential workflow to generate and refine a short blog post.

Prerequisites:
- Python 3.10–3.13
- pip install "crewai[tools]" openai
- OPENAI_API_KEY set in your environment
"""

import os
from textwrap import indent

from crewai import Agent, Task, Crew, Process  # core entities[web:75]
from crewai_tools import SerperDevTool  # optional web search tool (if configured)


# ---- 1. (Optional) Configure tools ----
# If you have SERPER_API_KEY set, we can enable a search tool to make
# the Researcher more realistic. Otherwise, we'll just skip it.
def get_research_tools():
  serper_key = os.getenv("SERPER_API_KEY")
  if serper_key:
    # SerperDevTool reads SERPER_API_KEY from env by default[web:72]
    return [SerperDevTool()]
  return []


# ---- 2. Define Agents ----
def build_researcher():
  return Agent(
    role="Researcher",
    goal=(
      "Gather accurate, up-to-date information about the requested topic "
      "and summarize the key points in a structured way."
    ),
    backstory=(
      "You are a meticulous research analyst who specializes in quickly "
      "surfacing trustworthy information and organizing it into clear notes."
    ),
    tools=get_research_tools(),
    verbose=True,
  )


def build_writer():
  return Agent(
    role="Writer",
    goal=(
      "Transform research notes into a clear, engaging, blog-style article "
      "for business and technical stakeholders."
    ),
    backstory=(
      "You are a senior content writer who explains complex AI topics in "
      "simple, practical language, with good structure and flow."
    ),
    verbose=True,
  )


def build_editor():
  return Agent(
    role="Editor",
    goal=(
      "Polish the draft for clarity, tone, and brevity while preserving "
      "technical accuracy."
    ),
    backstory=(
      "You are an experienced editor who tightens writing, removes repetition, "
      "and ensures the content is suitable for a professional audience."
    ),
    verbose=True,
  )


# ---- 3. Define Tasks ----
def build_tasks(topic: str, researcher: Agent, writer: Agent, editor: Agent):
  """Create three sequential tasks that pass context along the chain.[web:70]"""

  research_task = Task(
    description=(
      f"Research the topic: '{topic}'. "
      "Identify 5–8 key points, including definitions, benefits, challenges, "
      "and 1–2 simple enterprise examples. "
      "Output structured bullet-point notes."
    ),
    agent=researcher,
    expected_output=(
      "A bullet list of concise research notes covering definition, "
      "benefits, challenges, and examples."
    ),
  )

  writing_task = Task(
    description=(
      "Using the research notes provided in context, write a short blog-style "
      "article (400–600 words) explaining the topic to product managers and "
      "architects. Use clear headings and short paragraphs."
    ),
    agent=writer,
    context=[research_task],  # depends on research output[web:70]
    expected_output=(
      "A well-structured article with an introduction, 2–3 main sections, "
      "and a brief conclusion."
    ),
  )

  editing_task = Task(
    description=(
      "Review and edit the drafted article in the context. "
      "Improve clarity, remove repetition, fix grammar, and maintain a "
      "professional yet friendly tone. Keep the length similar."
    ),
    agent=editor,
    context=[writing_task],  # depends on writing output[web:70]
    expected_output=(
      "A polished version of the article, ready to be shared with stakeholders."
    ),
  )

  return [research_task, writing_task, editing_task]


# ---- 4. Build and Run the Crew ----
def run_content_creation_crew(topic: str) -> str:
  """Orchestrate the 3-agent workflow using a sequential process.[web:72]"""

  researcher = build_researcher()
  writer = build_writer()
  editor = build_editor()

  tasks = build_tasks(topic, researcher, writer, editor)

  crew = Crew(
    agents=[researcher, writer, editor],
    tasks=tasks,
    process=Process.sequential,  # research -> write -> edit[web:72]
    verbose=True,
  )

  result = crew.kickoff()  # executes tasks in order and returns final output[web:75]
  return result


if __name__ == "__main__":
  print("=" * 80)
  print(" CrewAI Demo: Researcher + Writer + Editor ")
  print("=" * 80)
  user_topic = input(
    "\nEnter a topic for the crew to write about "
    "(e.g., 'Agentic AI for IT incident management'): "
  ).strip()

  if not user_topic:
    user_topic = "Agentic AI in enterprise customer support"

  print(f"\n[INFO] Running crew for topic: {user_topic!r}\n")

  final_article = run_content_creation_crew(user_topic)

  print("\n" + "=" * 80)
  print(" FINAL POLISHED ARTICLE ")
  print("=" * 80)
  print(indent(str(final_article), prefix="  "))
  print("\nDone.")
