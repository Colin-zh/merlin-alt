import numpy as np
import pandas as pd
import lightgbm as lgb
import optuna
from sklearn.datasets import make_classification
from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.metrics import accuracy_score, roc_auc_score
from sklearn.preprocessing import StandardScaler
import warnings
warnings.filterwarnings('ignore')

class LightGBMOptunaCV:
    def __init__(self, n_trials=100, cv_folds=5, random_state=42):
        """
        初始化LightGBM + Optuna + CV训练器
        
        Parameters:
        - n_trials: Optuna优化次数
        - cv_folds: 交叉验证折数
        - random_state: 随机种子
        """
        self.n_trials = n_trials
        self.cv_folds = cv_folds
        self.random_state = random_state
        self.best_params = None
        self.best_score = None
        self.model = None
        self.study = None
        
    def objective(self, trial, X, y):
        """
        Optuna优化目标函数
        """
        # 定义超参数搜索空间
        param = {
            'objective': 'binary',
            'metric': 'binary_logloss',
            'verbosity': -1,
            'boosting_type': 'gbdt',
            'random_state': self.random_state,
            'n_estimators': trial.suggest_int('n_estimators', 100, 1000),
            'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.3),
            'num_leaves': trial.suggest_int('num_leaves', 20, 300),
            'max_depth': trial.suggest_int('max_depth', 3, 12),
            'min_child_samples': trial.suggest_int('min_child_samples', 5, 100),
            'subsample': trial.suggest_float('subsample', 0.5, 1.0),
            'colsample_bytree': trial.suggest_float('colsample_bytree', 0.5, 1.0),
            'reg_alpha': trial.suggest_float('reg_alpha', 1e-8, 10.0, log=True),
            'reg_lambda': trial.suggest_float('reg_lambda', 1e-8, 10.0, log=True),
        }
        
        # 设置交叉验证
        kf = StratifiedKFold(n_splits=self.cv_folds, shuffle=True, 
                            random_state=self.random_state)
        
        # 交叉验证计算平均分数
        cv_scores = []
        
        for train_idx, val_idx in kf.split(X, y):
            X_train, X_val = X[train_idx], X[val_idx]
            y_train, y_val = y[train_idx], y[val_idx]
            
            # 创建数据集
            train_data = lgb.Dataset(X_train, label=y_train)
            val_data = lgb.Dataset(X_val, label=y_val, reference=train_data)
            
            # 训练模型
            model = lgb.train(
                param,
                train_data,
                valid_sets=[val_data],
                callbacks=[
                    lgb.early_stopping(stopping_rounds=50, verbose=False),
                    lgb.log_evaluation(period=0)
                ]
            )
            
            # 预测并计算分数
            y_pred = model.predict(X_val)
            score = roc_auc_score(y_val, y_pred)
            cv_scores.append(score)
        
        # 返回平均AUC分数（Optuna会最大化这个分数）
        return np.mean(cv_scores)
    
    def fit(self, X, y):
        """
        训练模型
        
        Parameters:
        - X: 特征数据
        - y: 标签数据
        """
        print("开始超参数优化...")
        
        # 创建Optuna study
        self.study = optuna.create_study(
            direction='maximize',
            sampler=optuna.samplers.TPESampler(seed=self.random_state)
        )
        
        # 优化超参数
        self.study.optimize(
            lambda trial: self.objective(trial, X, y),
            n_trials=self.n_trials,
            show_progress_bar=True
        )
        
        # 获取最佳参数
        self.best_params = self.study.best_params
        self.best_score = self.study.best_value
        
        print(f"最佳超参数: {self.best_params}")
        print(f"最佳交叉验证分数: {self.best_score:.4f}")
        
        # 使用最佳参数训练最终模型
        print("训练最终模型...")
        final_params = self.best_params.copy()
        final_params.update({
            'objective': 'binary',
            'metric': 'binary_logloss',
            'verbosity': -1,
            'random_state': self.random_state
        })
        
        # 使用全部数据训练最终模型
        train_data = lgb.Dataset(X, label=y)
        self.model = lgb.train(
            final_params,
            train_data,
            num_boost_round=final_params['n_estimators']
        )
        
        return self
    
    def predict(self, X):
        """
        预测
        
        Parameters:
        - X: 特征数据
        
        Returns:
        - 预测概率
        """
        if self.model is None:
            raise ValueError("模型尚未训练，请先调用fit方法")
        return self.model.predict(X)
    
    def predict_proba(self, X):
        """
        预测概率（为了与sklearn接口一致）
        """
        predictions = self.predict(X)
        # 转换为二分类概率格式
        return np.column_stack([1 - predictions, predictions])
    
    def evaluate(self, X_test, y_test):
        """
        评估模型性能
        
        Parameters:
        - X_test: 测试特征
        - y_test: 测试标签
        """
        if self.model is None:
            raise ValueError("模型尚未训练，请先调用fit方法")
            
        y_pred = self.predict(X_test)
        y_pred_binary = (y_pred > 0.5).astype(int)
        
        accuracy = accuracy_score(y_test, y_pred_binary)
        auc = roc_auc_score(y_test, y_pred)
        
        print(f"测试集准确率: {accuracy:.4f}")
        print(f"测试集AUC: {auc:.4f}")
        
        return {'accuracy': accuracy, 'auc': auc}

# 示例使用
def main():
    # 生成示例数据
    print("生成示例数据...")
    X, y = make_classification(
        n_samples=1000,
        n_features=20,
        n_informative=15,
        n_redundant=5,
        n_clusters_per_class=1,
        random_state=42
    )
    
    # 数据标准化
    scaler = StandardScaler()
    X = scaler.fit_transform(X)
    
    # 划分训练测试集
    from sklearn.model_selection import train_test_split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    
    # 创建并训练模型
    model = LightGBMOptunaCV(n_trials=50, cv_folds=5, random_state=42)
    model.fit(X_train, y_train)
    
    # 评估模型
    print("\n模型评估:")
    model.evaluate(X_test, y_test)
    
    # 可视化Optuna优化过程
    try:
        import matplotlib.pyplot as plt
        
        # 优化历史
        fig = optuna.visualization.plot_optimization_history(model.study)
        fig.show()
        
        # 超参数重要性
        fig = optuna.visualization.plot_param_importances(model.study)
        fig.show()
        
    except ImportError:
        print("要可视化需要安装plotly: pip install plotly")


if __name__ == "__main__":
    main()