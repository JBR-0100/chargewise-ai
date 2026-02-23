import time
import random
import sys

banner = r"""
   ██████╗██╗  ██╗ █████╗ ██████╗  ██████╗ ███████╗██╗    ██╗██╗███████╗███████╗
  ██╔════╝██║  ██║██╔══██╗██╔══██╗██╔════╝ ██╔════╝██║    ██║██║██╔════╝██╔════╝
  ██║     ███████║███████║██████╔╝██║  ███╗█████╗  ██║ █╗ ██║██║███████╗█████╗  
  ██║     ██╔══██║██╔══██║██╔══██╗██║   ██║██╔══╝  ██║███╗██║██║╚════██║██╔══╝  
  ╚██████╗██║  ██║██║  ██║██║  ██║╚██████╔╝███████╗╚███╔███╔╝██║███████║███████╗
   ╚═════╝╚═╝  ╚═╝╚═╝  ╚═╝╚═╝  ╚═╝ ╚═════╝ ╚══════╝ ╚══╝╚══╝ ╚═╝╚══════╝╚══════╝
"""

for char in banner:
    sys.stdout.write(char)
    sys.stdout.flush()
    time.sleep(0.002)  # speed control

time.sleep(0.5)

messages = [
    "turning it off and on again...",
    "searching stack overflow...",
    "npm install (just in case)...",
    "fixing bugs you haven't written yet...",
    "compiling coffee into code...",
    "negotiating with semicolons...",
    "dividing by zero...",
    "removing console.log statements...",
    "adding console.log statements back...",
    "checking if it works on my machine...",
    "downloading more RAM...",
    "convincing ai to behave...",
    "summoning senior developer...",
    "pushing to production on Friday...",
    "blaming DNS...",
    "sacrificing a rubber duck...",
    "realizing it was a missing bracket...",
    "googling the error message...",
]

print("\nstarting chargewise ai...\n")
time.sleep(1)

for msg in random.sample(messages, 6):
    print(f"{msg} ", end="", flush=True)
    time.sleep(random.uniform(0.7, 1.4))
    print(" done")

time.sleep(0.5)

# ---------------- Dependency Check ----------------
print("\nchecking environment...\n")

requirements = {
    "pandas": "2.0",
    "numpy": "1.26",
    "sklearn": "1.4",
    "matplotlib": "3.8",
    "seaborn": "0.13",
    "streamlit": "1.32",
}

results = []

def version_ok(installed, required):
    return tuple(map(int, installed.split(".")[:2])) >= tuple(map(int, required.split(".")))

# pandas
try:
    import pandas as pd
    ok = version_ok(pd.__version__, requirements["pandas"])
    pd.DataFrame({"a": [1, 2]}).sum()
    results.append(("pandas", pd.__version__, ok))
except Exception:
    results.append(("pandas", "missing", False))

# numpy
try:
    import numpy as np
    ok = version_ok(np.__version__, requirements["numpy"])
    np.array([1, 2, 3]).mean()
    results.append(("numpy", np.__version__, ok))
except Exception:
    results.append(("numpy", "missing", False))

# scikit-learn
try:
    from sklearn.linear_model import LinearRegression
    import sklearn
    ok = version_ok(sklearn.__version__, requirements["sklearn"])
    LinearRegression().fit([[0], [1]], [0, 1])
    results.append(("scikit-learn", sklearn.__version__, ok))
except Exception:
    results.append(("scikit-learn", "missing", False))

# matplotlib
try:
    import matplotlib
    ok = version_ok(matplotlib.__version__, requirements["matplotlib"])
    results.append(("matplotlib", matplotlib.__version__, ok))
except Exception:
    results.append(("matplotlib", "missing", False))

# seaborn
try:
    import seaborn as sns
    ok = version_ok(sns.__version__, requirements["seaborn"])
    results.append(("seaborn", sns.__version__, ok))
except Exception:
    results.append(("seaborn", "missing", False))

# streamlit
try:
    import streamlit as st
    ok = version_ok(st.__version__, requirements["streamlit"])
    results.append(("streamlit", st.__version__, ok))
except Exception:
    results.append(("streamlit", "missing", False))

# ---------------- report ----------------
all_ok = True

for name, version, ok in results:
    status = "OK" if ok else "FAIL"
    print(f"{name:<14} : {version:<10} {status}")
    if not ok:
        all_ok = False

time.sleep(0.5)

# ---------------- final message ----------------
if all_ok:
    print("\nsystem ready.")
    print("welcome to chargewise ai.")
    print("environment check passed.")
else:
    print("\nwarning: some dependencies are missing or outdated.")
    print("run: pip install -r requirements.txt\n")