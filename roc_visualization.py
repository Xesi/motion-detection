# %%
import pandas as pd 
import plotly.express as px

def get_df(filename):
    
    video = []
    TPR = []
    FPR = []
    treshold = []
    
    with open(filename) as f:
        lines = f.readlines()
        data = [l.split(',') for l in lines]
        video = [d[0] for d in data]
        treshold = [int(d[1]) for d in data]
        #acc = [float(d[2]) for d in data]
        TPR = [float(d[3]) for d in data]
        FPR = [float(d[4]) for d in data]
        
    method_name = filename.split('_')[1].split('.')[0]
    method = [method_name + ' ' + video[i] for i in range(len(video))]
    
    data = {
        'video': video,
        'treshold': treshold,
        'TPR': TPR,
        'FPR': FPR,
        'method': method
    }

    return pd.DataFrame(data)

df = get_df("result_method1.txt")

# print(FPR)

fig = px.line(df, x="FPR", y="TPR", color="video", hover_name="treshold", markers=True)
fig.show()

# %%
import pandas as pd 
import plotly.express as px


def get_df(filename):
    
    video = []
    TPR = []
    FPR = []
    treshold = []
    
    with open(filename) as f:
        lines = f.readlines()
        data = [l.split(',') for l in lines]
        video = [d[0] for d in data]
        treshold = [int(d[1]) for d in data]
        #acc = [float(d[2]) for d in data]
        TPR = [float(d[3]) for d in data]
        FPR = [float(d[4]) for d in data]
        
    method_name = filename.split('_')[1].split('.')[0]
    method = [method_name + ' ' + video[i] for i in range(len(video))]
    
    data = {
        'video': video,
        'treshold': treshold,
        'TPR': TPR,
        'FPR': FPR,
        'method': method
    }

    return pd.DataFrame(data)
    

df1 = get_df("result_method1.txt")
df2 = get_df("result_method2.txt")

df = pd.concat([df1, df2], ignore_index=True, sort=False)



# print(FPR)

fig = px.line(df, x="FPR", y="TPR", color="method", hover_name="treshold", markers=True)
fig.show()

# %%
df = get_df("result_method3.txt")
fig = px.line(df, x="FPR", y="TPR", color="method", hover_name="treshold", markers=True)
fig.show()

# %%
df = get_df("result_method4.txt")
fig = px.line(df, x="FPR", y="TPR", color="method", hover_name="treshold", markers=True)
fig.show()

# %%



