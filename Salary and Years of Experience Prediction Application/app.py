import joblib 
import numpy as np 

application_name = "Salary Prediction Application"
options = """Options:
1. Predict the Salary 
2. Predict the Year of Experience
3. Exit
"""
salary_model = joblib.load("./models/salary.pkl")
yoe_model = joblib.load("./models/YearsExperience.pkl")

while True:
    print(application_name)
    print(options)
    choice = int(input("Enter your choice: "))
    match choice:
        case 1:
            print("Predicting Salary:")
            yoe = float(input("Enter Years of Experience: "))
            yoe = np.array(yoe).reshape(-1,1)
            salary = salary_model.predict(yoe)
            print(f"Predicted Salary: {int(salary[0][0])}")

        case 2:
            print("Predicting Year of Experience: ")
            salary = int(input("Enter your Expected Salary: "))
            salary = np.array(salary)
            salary = salary.reshape(-1,1)
            yoe = yoe_model.predict(salary)
            print(f"Predicted Years of Experience: {yoe[0][0]:.1f}")
        case 3:
            print("Thank You, Shutting down...")
            break 
