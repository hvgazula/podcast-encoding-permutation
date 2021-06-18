import pandas as pd

bobbi = pd.read_csv('phase-5000-sig-elec-gpt2xl50d-perElec-FDR-01-LH.csv', header=None)[0].tolist()
# harsha_df = pd.read_csv('harsha_by.csv')


def function1(x):
    split_list = x.split('conversation1')
    electrode = split_list[-1].strip('_')
    subject = int(split_list[0][2:5])
    return (subject, electrode)


sigelec_list = [function1(item) for item in bobbi]
bobbi_df = pd.DataFrame(sigelec_list, columns=['subject', 'electrode'])

bobbi = [tuple(x) for x in bobbi_df.values]
# harsha = [tuple(x) for x in harsha_df.values]

# not_in_bobbi = pd.DataFrame(set(harsha) - set(bobbi), columns=['subject', 'electrode'])
# not_in_harsha = pd.DataFrame(set(bobbi) - set(harsha), columns=['subject', 'electrode'])
# intersection = pd.DataFrame(set(bobbi) & set(harsha), columns=['subject', 'electrode'])
# union = pd.DataFrame(set(bobbi) | set(harsha), columns=['subject', 'electrode'])

bobbi_df.to_csv('bobbi_new.csv', index=False)
# not_in_bobbi.to_csv('not_in_bobbi_by.csv', index=False)
# not_in_harsha.to_csv('not_in_harsha_by.csv', index=False)
# intersection.to_csv('intersection_by.csv', index=False)
# union.to_csv('union_by.csv', index=False)
