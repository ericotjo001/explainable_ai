DESCRIPTION = """
python main.py
python main.py --mode data
python main.py --mode training
python main.py --mode evaluation
python main.py --mode workflow
"""

NULL_DESCRIPTION = """Invalid or no mode selected. Rerouting..."""

WORKFLOW_DESCRIPTION = """
python main.py
python main.py --mode workflow
python main.py --mode workflow --mode2 workflow1
python main.py --mode workflow --mode2 workflow2
python main.py --mode workflow --mode2 workflow3
python main.py --mode workflow --mode2 workflow4"""