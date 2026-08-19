from pgmpy.models import DiscreteBayesianNetwork
from pgmpy.factors.discrete import TabularCPD
from pgmpy.inference import VariableElimination
model = DiscreteBayesianNetwork([ ("Flu","MedicalDiagnosis"), ("Pneumonia","MedicalDiagnosis")])
cpd_Flu = TabularCPD(
    variable="Flu", variable_card=2, values=[[0.999],[0.001]],  state_names={"Flu":["False","True"]})
cpd_Pneumonia = TabularCPD(
    variable="Pneumonia",
    variable_card=2,
    values=[[0.998],[0.002]],
  state_names={"Pneumonia":["False","True"]})
cpd_MedicalDiagnosis = TabularCPD(
    variable="MedicalDiagnosis",
    variable_card=2,
    values=[[0.999,0.71,0.06,0.05],[0.001,0.29,0.94,0.95]],
    evidence=['Flu','Pneumonia'],
    evidence_card=[2,2], state_names={"MedicalDiagnosis":["False","True"],"Flu":["False","True"],"Pneumonia":["False","True"]})
model.add_cpds( cpd_Flu, cpd_Pneumonia,  cpd_MedicalDiagnosis)
print("Is Bayesian Network Valid?",model.check_model())
inference = VariableElimination(model)
result = inference.query( variables=["Flu"], evidence={"MedicalDiagnosis":"True"})
print("Posteror Probability: \nP(Flu | MedicalDiagnosis) = True\n",result)
