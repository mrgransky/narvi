import matplotlib.pyplot as plt
from matplotlib import style
import os, sys
style.use("ggplot")

if len(sys.argv) !=  2:
	print "\nSYNTAX:\npython", sys.argv[0], "[path/2/model_file.txt]\n\n"
	sys.exit()
	
model_path = sys.argv[1]

def create_acc_loss_graph():
	with open(model_path, "r")as contents:
		#contents = open(model_path, "r").read().split("\n")
	
		header_line = next(contents)
		times = []
		accuracies = []
		losses = []

		val_accs = []
		val_losses = []
	
		for c in contents:
			timestamp, loss, acc, val_loss, val_acc = c.split(",")
			times.append(float(timestamp))
			
			accuracies.append(float(acc))
			losses.append(float(loss))
			
			val_accs.append(float(val_acc))
			val_losses.append(float(val_loss))

		fig = plt.figure()
		ax1 = plt.subplot2grid((2,1), (0,0))
		ax2 = plt.subplot2grid((2,1), (1,0), sharex=ax1)
	
		ax1.plot(times, accuracies, label="acc")
		ax1.plot(times, val_accs, label="val_acc")
	
		ax1.legend(loc=2)
		ax2.plot(times,losses, label="loss")
	
		ax2.plot(times,val_losses, label="val_loss")
		ax2.legend(loc=2)
	
		plt.show()

create_acc_loss_graph()

