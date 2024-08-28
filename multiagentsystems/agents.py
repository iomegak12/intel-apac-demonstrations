from crewai import Agent
from crewai_tools import WebsiteSearchTool, SerperDevTool, FileReadTool

web_search_tool = WebsiteSearchTool()
serper_dev_tool = SerperDevTool()
file_read_tool = FileReadTool(
    file_path="./job_description_example.md",
    description="A tool that reads job description example file"
)


class MyAgents():
    def research_agent(self):
        return Agent(
            role="Research Analyst",
            goal="Analyze the company website and provided description to extract insights on culture, values and specific needs",
            tools=[web_search_tool, serper_dev_tool],
            backstory="Expert in analyzing company cultures and identifying key values and needs from various sources including websites and brief descriptionds",
            verbose=True
        )

    def writer_agent(self):
        return Agent(
            role="Job Description Writer",
            goal="Use insights from the research analyst to create a detailed, engaging and awesome job descriptions for job posting ",
            tools=[web_search_tool, serper_dev_tool, file_read_tool],
            backstory="Skilled in crafting a compelling job description that outstand or resonate with the company values and attract the right candidates",
            verbose=True
        )

    def reviewer_agent(self):
        return Agent(
            role="Review and Editing Specialist",
            goal="Review the job posting for clarity, engagement and grammatical accuracy and alignment with company values",
            tools=[web_search_tool, serper_dev_tool, file_read_tool],
            backstory="A strict and meticulous editor with an eye for detail, ensuring that every piece of content is clear, engaging and grammatically perfect.",
            verbose=True
        )
