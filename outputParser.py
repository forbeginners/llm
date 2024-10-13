from langchain_core.output_parsers import PydanticOutputParser
from pydantic import BaseModel, Field


class Tag(BaseModel):
    tag: str = Field(description="Name of the tag")


class Planner(BaseModel):
    todos: list[str] = Field(description="todos list")


class GoalKeyword(BaseModel):
    goalKeyword: list[str] = Field(description="Goal Keyword")


class ActivityKeyword(BaseModel):
    activityKeyword: list[str] = Field(description="Activity Keyword")

class Category(BaseModel):
    categories: list[str] = Field(description="Category")

class ExtractResult(BaseModel):
    goalKeyword: str = Field(description="goalKeyword Keyword")
    activityKeyword: str = Field(description="activityKeyword Keyword")


planner_parser = PydanticOutputParser(pydantic_object=Planner)
tag_parser = PydanticOutputParser(pydantic_object=Tag)
activityKeyword_parser = PydanticOutputParser(pydantic_object=ActivityKeyword)
goalKeyword_parser = PydanticOutputParser(pydantic_object=GoalKeyword)
extract_result_parser = PydanticOutputParser(pydantic_object=ExtractResult)
extract_categories_parser = PydanticOutputParser(pydantic_object=Category)