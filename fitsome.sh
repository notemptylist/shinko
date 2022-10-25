git pull --rebase;
python -u fit_arima.py -s yarx_vmw.json && git add modelfits ;
git commit -m 'New Params';
git pull --rebase; 
git push 
