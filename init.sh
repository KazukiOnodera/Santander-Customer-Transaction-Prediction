git clone git@github.com:KazukiOnodera/santander-customer-transaction-prediction.git

cd santander-customer-transaction-prediction
mkdir input
mkdir output
mkdir data
mkdir feature
mkdir py
mkdir jn
mkdir py/LOG
cd input
kaggle competitions download -c santander-customer-transaction-prediction
cd ../
echo *.DS_Store > .gitignore
echo ~$*.xls* >> .gitignore
echo feature/ >> .gitignore
echo input/ >> .gitignore
echo output/ >> .gitignore
echo data/ >> .gitignore
echo external/ >> .gitignore
echo jn/.ipynb_checkpoints >> .gitignore
echo py/*.model >> .gitignore
echo py/*.p >> .gitignore
echo py/__pycache__/* >> .gitignore
echo py/~$*.xls* >> .gitignore
cat .gitignore

gitupdate

