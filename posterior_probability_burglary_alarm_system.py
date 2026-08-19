from pgmpy.models import DiscreteBayesianNetwork
from pgmpy.factors.discrete import TabularCPD
from pgmpy.inference import VariableElimination

model = DiscreteBayesianNetwork([
    ("Burglary","Alarm"),
    ("Earthquake","Alarm")
])

cpd_burglary = TabularCPD(
    variable="Burglary",
    variable_card=2,
    values=[[0.999],[0.001]],
    state_names={"Burglary":["False","True"]}
)

cpd_earthquake = TabularCPD(
    variable="Earthquake",
    variable_card=2,
    values=[[0.998],[0.002]],
    state_names={"Earthquake":["False","True"]}
)


cpd_alarm = TabularCPD(
    variable="Alarm",
    variable_card=2,
    values=[[0.999,0.71,0.06,0.05],[0.001,0.29,0.94,0.95]],
    evidence=['Burglary','Earthquake'],
    evidence_card=[2,2],
    state_names={"Alarm":["False","True"],"Burglary":["False","True"],"Earthquake":["False","True"]}
)

model.add_cpds(
    cpd_burglary,
    cpd_earthquake,
    cpd_alarm
)

print("Is Bayesian Network Valid?",model.check_model())
inference = VariableElimination(model)
result = inference.query(
    variables=["Burglary"],
    evidence={"Alarm":"True"}
)

print("Posteror Probability: \nP(Burglary | Alarm = True)\n",result)
