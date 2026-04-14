import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler

# 1. إعداد البيانات (افترضنا أنك قمت بتنظيف البيانات مسبقاً)
# X = df.drop('Churn', axis=1)
# y = df['Churn']

# 2. ضروري جداً: توحيد المقاييس (Standardization) لجعل الـ coefficients قابلة للمقارنة
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# 3. توليد 20 قيمة لـ C بشكل لوغاريتمي (من 0.001 إلى 100)
C_values = np.logspace(-3, 2, 20)

penalties = ['l1', 'l2']
coef_data = {p: [] for p in penalties}

# 4. تدريب النماذج وتسجيل المعاملات
for p in penalties:
    for c in C_values:
        # ملاحظة: solver='saga' يدعم L1 و L2 وهو سريع مع المجموعات الكبيرة
        model = LogisticRegression(penalty=p, C=c, solver='saga', max_iter=5000)
        model.fit(X_scaled, y)
        coef_data[p].append(model.coef_[0])

# 5. التصوير البياني (Visualization)
fig, axes = plt.subplots(1, 2, figsize=(16, 6), sharey=True)

for i, p in enumerate(penalties):
    coefs = np.array(coef_data[p])
    for j in range(coefs.shape[1]):
        axes[i].plot(C_values, coefs[:, j], label=X.columns[j] if i==0 else "")
    
    axes[i].set_xscale('log')
    axes[i].set_title(f'Regularization Path: {p.upper()}')
    axes[i].set_xlabel('C (Inverse Regularization Strength)')
    axes[i].grid(True, linestyle='--', alpha=0.6)

axes[0].set_ylabel('Coefficient Magnitude')
fig.legend(loc='center right', bbox_to_anchor=(1.1, 0.5))
plt.tight_layout()
plt.show()
