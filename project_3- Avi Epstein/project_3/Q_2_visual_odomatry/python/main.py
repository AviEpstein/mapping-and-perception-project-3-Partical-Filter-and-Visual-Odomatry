from project_questions import ProjectQuestions


if __name__ == "__main__":
    vo_data = {}
    vo_data['dir'] = r"C:\dataset"
    vo_data['sequence'] = 0
    
    project = ProjectQuestions(vo_data)
    project.run()