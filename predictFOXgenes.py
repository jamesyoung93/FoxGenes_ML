

import argparse, os, warnings, joblib
import numpy as np, pandas as pd
from tqdm import tqdm
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from xgboost import XGBClassifier
import statsmodels.api as sm
from numpy.linalg import LinAlgError

FLAG_COL = "divergent_promoter_distance"   # always shown in coefficient table

# ───────── helper utilities ─────────
def add_flags(df, cols):
    for c in cols:
        inf = np.isinf(df[c]); na = df[c].isna()
        if inf.any():
            df[f"{c}__was_inf"] = inf.astype(int); df.loc[inf, c] = 0
        if na.any():
            df[f"{c}__was_na"]  = na.astype(int); df.loc[na, c]  = 0
    return df

def drop_constants(df, cols, keep=()):
    return [c for c in cols if (c in keep) or df[c].nunique(dropna=False) > 1]

def corr_filter(df, thr=0.95, protect=()):
    corr = np.abs(np.corrcoef(df.T))
    upper = np.triu(np.ones_like(corr, bool), 1)
    drop = set(df.columns[np.where((corr > thr) & upper)[1]]) - set(protect)
    return [c for c in df.columns if c not in drop]

def encode_labels(s):
    txt=s.astype(str).str.lower()
    arr=np.full(len(s),-1,dtype=int)
    arr[txt=="fox"]=1
    arr[txt.isin(["notfox","nonfox","not fox","not_fox"])]=0
    return pd.Series(arr,index=s.index)

def stepwise_select(X,y,thr_in=.01,thr_out=.05):
    inc=[]
    while True:
        changed=False
        exc=list(set(X.columns)-set(inc))
        best_p,best_f=1.0,None
        for f in exc:
            try:p=sm.Logit(y,sm.add_constant(X[inc+[f]])).fit(disp=0).pvalues[f]
            except LinAlgError:p=1.0
            if p<best_p:best_p,best_f=p,f
        if best_p<thr_in:inc.append(best_f);changed=True
        if inc:
            try:pvals=sm.Logit(y,sm.add_constant(X[inc])).fit(disp=0).pvalues.iloc[1:]
            except LinAlgError:pvals=pd.Series(0,index=inc)
            worst=pvals.max()
            if worst>thr_out:inc.remove(pvals.idxmax());changed=True
        if not changed:break
    return inc

def fi_df(model,names):
    if hasattr(model,"coef_"):imp=np.abs(model.coef_).ravel()
    elif hasattr(model,"feature_importances_"):imp=model.feature_importances_
    else:return pd.DataFrame()
    return pd.DataFrame({"feature":names,"importance":imp}).sort_values("importance",ascending=False)

def greedy_pack(df,kb):
    df=df.sort_values("score_per_kb",ascending=False)
    sel,used=[],0
    for _,r in df.iterrows():
        if used+r["gene_len_kb"]<=kb:sel.append(r);used+=r["gene_len_kb"]
    return pd.DataFrame(sel)

# ───────── main ─────────
def main(a):
    os.makedirs(a.out,exist_ok=True)
    df=pd.read_csv(a.data)

    # numeric tidy
    num_cols=[c for c in df.select_dtypes(include=[np.number]).columns if c!=a.label_col]
    df=add_flags(df,num_cols)
    num_cols=drop_constants(df,num_cols,keep=[FLAG_COL])
    if FLAG_COL not in num_cols and FLAG_COL in df.columns:num_cols.append(FLAG_COL)
    df=df[[a.id_col,a.label_col]+num_cols]
    print(f"[INFO] After constant-col drop → {df.shape}")

    y_full=encode_labels(df[a.label_col]); mask_lab=y_full.isin([0,1]); mask_unk=y_full==-1
    cand_cols=corr_filter(df[num_cols],0.95,protect=[FLAG_COL])
    X_cand, y_lab=df.loc[mask_lab,cand_cols], y_full[mask_lab]

    logreg_cols_opt=stepwise_select(X_cand,y_lab)
    print(f"[INFO] LogReg (opt) uses {len(logreg_cols_opt)} features")

    # for coefficient table
    logreg_cols_coef=logreg_cols_opt.copy()
    if FLAG_COL in df.columns and FLAG_COL not in logreg_cols_coef:
        logreg_cols_coef.append(FLAG_COL)

    X_full_lab=df.loc[mask_lab,num_cols]
    X_red_lab=df.loc[mask_lab,logreg_cols_opt]

    # model zoo (opt logreg)
    models={
        "logreg":Pipeline([("scale",StandardScaler()),
                           ("clf",LogisticRegression(max_iter=2000,
                                                     class_weight="balanced",
                                                     n_jobs=-1))]),
        "rf":RandomForestClassifier(n_estimators=300,n_jobs=-1,
                                    class_weight="balanced",random_state=42),
        "xgb":XGBClassifier(n_estimators=200,learning_rate=0.05,max_depth=6,
                            subsample=0.9,colsample_bytree=0.8,
                            objective="binary:logistic",eval_metric="logloss",
                            n_jobs=max(1,os.cpu_count()-1),random_state=42)
    }

    # CV
    from joblib import Parallel,delayed; rng=np.random.default_rng(42)
    splits=[(f,rng.integers(1_000_000)) for f in a.frac for _ in range(a.reps)]
    def cv_one(frac,seed):
        sss=StratifiedShuffleSplit(n_splits=1,train_size=frac,random_state=seed)
        tr,te=next(sss.split(X_full_lab,y_lab)); row,preds={"train_frac":frac},[]
        for n,m in models.items():
            Xtr=X_red_lab.iloc[tr] if n=="logreg" else X_full_lab.iloc[tr]
            Xte=X_red_lab.iloc[te] if n=="logreg" else X_full_lab.iloc[te]
            ytr,yte=y_lab.iloc[tr],y_lab.iloc[te]
            m.fit(Xtr,ytr)
            preds.append(m.predict_proba(Xte)[:,1])
            row[f"{n}_train_auc"]=roc_auc_score(ytr,m.predict_proba(Xtr)[:,1])
            row[f"{n}_test_auc"] =roc_auc_score(yte,preds[-1])
        row["ensemble_test_auc"]=roc_auc_score(yte,np.mean(preds,axis=0))
        return row
    print(f"[INFO] Running {len(splits)} CV splits …")
    pd.DataFrame(Parallel(n_jobs=a.njobs)(delayed(cv_one)(f,s) for f,s in tqdm(splits))
                 ).to_csv(f"{a.out}/metrics.csv",index=False)

    # final predictive fits
    fitted={}
    for n,m in models.items():
        m.fit(X_red_lab if n=="logreg" else X_full_lab,y_lab)
        fitted[n]=m; joblib.dump(m,f"{a.out}/{n}_model.pkl")
        fi=fi_df(m, logreg_cols_opt if n=="logreg" else num_cols)
        if not fi.empty: fi.to_csv(f"{a.out}/{n}_feature_importance.csv",index=False)

    # coefficient/p-value table with FLAG forced
    X_sm = sm.add_constant(StandardScaler().fit_transform(df.loc[mask_lab,logreg_cols_coef]))
    sm_res = sm.Logit(y_lab, X_sm).fit(disp=False)
    pd.DataFrame({"feature":["const"]+logreg_cols_coef,
                  "coef":sm_res.params,
                  "pvalue":sm_res.pvalues}) \
       .to_csv(f"{a.out}/logreg_coef_pvalues.csv",index=False)

    # probabilities
    X_full=df[num_cols]; X_red=df[logreg_cols_opt]
    probs=np.column_stack([fitted["logreg"].predict_proba(X_red)[:,1],
                           fitted["rf"].predict_proba(X_full)[:,1],
                           fitted["xgb"].predict_proba(X_full)[:,1]])
    df_pred=(df[[a.id_col,a.label_col]]
             .assign(logreg_prob=probs[:,0],rf_prob=probs[:,1],
                     xgb_prob=probs[:,2],ensemble_prob=probs.mean(axis=1)))
    df_pred.to_csv(f"{a.out}/probabilities.csv",index=False)

    # greedy complement
    len_col="gene_len_kb" if "gene_len_kb" in df else "gene_length" if "gene_length" in df else None
    if len_col and mask_unk.any():
        cand=(df_pred.loc[mask_unk,[a.id_col,"ensemble_prob"]]
              .merge(df[[a.id_col,len_col]],on=a.id_col))
        cand["gene_len_kb"]=cand[len_col]/1_000
        cand["score_per_kb"]=cand["ensemble_prob"]/cand["gene_len_kb"]
        greedy_pack(cand,a.kb_limit).to_csv(f"{a.out}/greedy_complement.csv",index=False)

    print("✓ Finished – outputs in", os.path.abspath(a.out))

# ───── CLI
if __name__=="__main__":
    warnings.filterwarnings("ignore", category=RuntimeWarning, module="statsmodels")
    ap=argparse.ArgumentParser()
    ap.add_argument("--data",required=True)
    ap.add_argument("--id_col",default="gene")
    ap.add_argument("--label_col",default="label")
    ap.add_argument("--out",default="fox_outputs")
    ap.add_argument("--reps",default=20,type=int)
    ap.add_argument("--frac",nargs="+",type=float,default=[0.7,0.8,0.9])
    ap.add_argument("--kb_limit",default=100,type=int)
    ap.add_argument("--njobs",default=os.cpu_count(),type=int)
    main(ap.parse_args())
