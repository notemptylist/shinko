git pull --rebase;
python -u fit_arima.py && git add modelfits ;
git commit -m 'New Params';
git pull --rebase; 
git push 
