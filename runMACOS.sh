read -p "Process will be resource heavy and may up to few minutes to finish. Continue? [y|n]" decision
if [ $decision = "y" ]
then
	sudo brew update > /dev/null
	sudo brew install python > /dev/null
	pip3 install -r requirements.txt > /dev/null
	python3 predict_salary.py
else
	echo "Bye bye!"
fi