//Burglary Alarm System
%pip install pgmpy
import numpy as np
from pgmpy.models import DiscreteBayesianNetwork
from pgmpy.factors.discrete import TabularCPD
from pgmpy.inference import VariableElimination

model = DiscreteBayesianNetwork([
    ("Burglary", "Alarm"),
    ("Earthquake", "Alarm"),
    ("Alarm", "JohnCalls"),
    ("Alarm", "SarahCalls")
])

cpd_burglary = TabularCPD(
    variable="Burglary",
    variable_card=2,
    values=np.array([[0.999], [0.001]]),
    state_names={"Burglary": ["False", "True"]}
)

cpd_earthquake = TabularCPD(
    variable="Earthquake",
    variable_card=2,
    values=np.array([[0.998], [0.002]]),
    state_names={"Earthquake": ["False", "True"]}
)

cpd_alarm = TabularCPD(
    variable="Alarm",
    variable_card=2,
    values=np.array([
        [0.999, 0.71, 0.06, 0.05],
        [0.001, 0.29, 0.94, 0.95]
    ]),
    evidence=["Burglary", "Earthquake"],
    evidence_card=[2, 2],
    state_names={
        "Alarm": ["False", "True"],
        "Burglary": ["False", "True"],
        "Earthquake": ["False", "True"]
    }
)

cpd_john = TabularCPD(
    variable="JohnCalls",
    variable_card=2,
    values=np.array([
        [0.95, 0.10],
        [0.05, 0.90]
    ]),
    evidence=["Alarm"],
    evidence_card=[2],
    state_names={
        "JohnCalls": ["False", "True"],
        "Alarm": ["False", "True"]
    }
)

cpd_sarah = TabularCPD(
    variable="SarahCalls",
    variable_card=2,
    values=np.array([
        [0.99, 0.30],
        [0.01, 0.70]
    ]),
    evidence=["Alarm"],
    evidence_card=[2],
    state_names={
        "SarahCalls": ["False", "True"],
        "Alarm": ["False", "True"]
    }
)

model.add_cpds(
    cpd_burglary,
    cpd_earthquake,
    cpd_alarm,
    cpd_john,
    cpd_sarah
)

print("Is Bayesian Network Valid?", model.check_model())

inference = VariableElimination(model)
result = inference.query(
    variables=["Burglary"],
    evidence={
        "JohnCalls": "True",
        "SarahCalls": "True"
    }
)

print("Probability of Burglary given JohnCalls=True and SarahCalls=True")
print(result)
