# Medical Diagnosis
import numpy as np
from pgmpy.models import DiscreteBayesianNetwork
from pgmpy.factors.discrete import TabularCPD
from pgmpy.inference import VariableElimination

model = DiscreteBayesianNetwork([
    ("Flu", "MedicalDiagnosis"),
    ("Pneumonia", "MedicalDiagnosis"),
    ("MedicalDiagnosis", "Fever"),
    ("MedicalDiagnosis", "Cough")
])

cpd_Flu = TabularCPD(
    variable="Flu",
    variable_card=2,
    values=np.array([[0.999], [0.001]]),
    state_names={"Flu": ["False", "True"]}
)

cpd_earthquake = TabularCPD(
    variable="Pneumonia",
    variable_card=2,
    values=np.array([[0.998], [0.002]]),
    state_names={"Pneumonia": ["False", "True"]}
)

cpd_MedicalDiagnosis = TabularCPD(
    variable="MedicalDiagnosis",
    variable_card=2,
    values=np.array([
        [0.998, 0.75, 0.05, 0.10],
        [0.002, 0.25, 0.95, 0.90]
    ]),
    evidence=["Flu", "Pneumonia"],
    evidence_card=[2, 2],
    state_names={
        "MedicalDiagnosis": ["False", "True"],
        "Flu": ["False", "True"],
        "Pneumonia": ["False", "True"]
    }
)

cpd_john = TabularCPD(
    variable="Fever",
    variable_card=2,
    values=np.array([
        [0.97, 0.25],
        [0.03, 0.75]
    ]),
    evidence=["MedicalDiagnosis"],
    evidence_card=[2],
    state_names={
        "Fever": ["False", "True"],
        "MedicalDiagnosis": ["False", "True"]
    }
)

cpd_sarah = TabularCPD(
    variable="Cough",
    variable_card=2,
    values=np.array([
        [0.95, 0.40],
        [0.05, 0.60]
    ]),
    evidence=["MedicalDiagnosis"],
    evidence_card=[2],
    state_names={
        "Cough": ["False", "True"],
        "MedicalDiagnosis": ["False", "True"]
    }
)

model.add_cpds(
    cpd_Flu,
    cpd_earthquake,
    cpd_MedicalDiagnosis,
    cpd_john,
    cpd_sarah
)

print("Is Bayesian Network Valid?", model.check_model())

inference = VariableElimination(model)
result = inference.query(
    variables=["Flu"],
    evidence={
        "Fever": "True",
        "Cough": "True"
    }
)

print("Probability of Flu given Fever=True and Cough=True")
print(result)
print("HANOCH SHETTY T043")
