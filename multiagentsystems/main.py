import os

from dotenv import load_dotenv
from crewai import Crew
from tasks import MyTasks
from agents import MyAgents


def main():
    load_dotenv()

    tasks = MyTasks()
    agents = MyAgents()

    company_description = input("what's the company description?\n")
    company_domain = input("What's the company Domain URL?\n")
    hiring_needs = input("What are all the hiring needs?\n")
    specific_benefits = input("what are all the specific benefits you offer?")

    research_agent = agents.research_agent()
    writer_agent = agents.writer_agent()
    reviewer_agent = agents.reviewer_agent()

    research_company_task = tasks.research_company_culture_task(research_agent, company_description, company_domain)
    industry_analysis_task = tasks.industry_analysis_task(research_agent, company_domain)
    research_role_requirements_task = tasks.research_role_requirements_task(research_agent, hiring_needs)
    draft_job_posting_task = tasks.draft_job_posting_task(writer_agent, company_description, hiring_needs, specific_benefits)
    review_edit_job_posting_task = tasks.review_and_edit_job_posting_task(reviewer_agent, hiring_needs)

    process_crew = Crew(
        agents = [research_agent, writer_agent, reviewer_agent],
        tasks = [
            research_company_task,
            research_role_requirements_task,
            industry_analysis_task,
            draft_job_posting_task,
            review_edit_job_posting_task
        ]
    )

    result = process_crew.kickoff()

    print("Job Posting Creation Process is Completed ...")
    print("Final Job Posting ...")

    print(result)


if __name__ == "__main__":
    try:
        main()
    except Exception as error:
        print(f"Error Occurred, Details : {error}")
