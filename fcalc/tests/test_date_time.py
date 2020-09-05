from datetime import date
from datetime import datetime
# today = date.today()
# d1 = today.strftime("%d/%m/%Y")
# d2 = today.strftime("%B %d, %Y")
now = datetime.now()
dt_string = now.strftime("%Y_%m_%d_%H_%M_%S")
# print("d1 =", d1)
# print("d2 =", d2)
# print("now =", now)
print("date and time =", dt_string)	
