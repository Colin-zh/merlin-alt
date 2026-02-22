from typing import Dict, List, Any, Optional, Union, Callable
import warnings
warnings.filterwarnings('ignore')

import numpy as np
import pandas as pd
from sklearn.datasets import make_classification
from sklearn.model_selection import StratifiedKFold, cross_val_score, train_test_split
from sklearn.metrics import accuracy_score, roc_auc_score
from sklearn.preprocessing import StandardScaler

import lightgbm as lgb
import optuna


class LightGBMOptunaCV:
    def __init__(self, 
                 n_trials: int = 100, 
                 param_space: Optional[Dict[str, Any]] = None,
                 cv_folds: int = 5,
                 enable_cv: bool = True,
                 random_state: int = 42):
        """
        初始化LightGBM + Optuna训练器
        
        Parameters:
        - n_trials: Optuna优化次数
        - param_space: 自定义超参数搜索空间
            - int: {'type': 'int', 'low': int, 'high': int}
            - float: {'type': 'float', 'low': float, 'high': float, 'log': bool}
            - categorical: {'type': 'categorical', 'choices': List[Any]}
            - fixed: {'type': 'fixed', 'value': Any}
        - cv_folds: 交叉验证折数（当enable_cv=True时有效）
        - enable_cv: 是否启用交叉验证
        - random_state: 随机种子
        """
        self.n_trials = n_trials
        self.cv_folds = cv_folds
        self.enable_cv = enable_cv
        self.random_state = random_state
        self.best_params = None
        self.best_score = None
        self.model = None
        self.study = None
        self.scaler = None
        
        # 设置默认超参数搜索空间
        self.default_param_space = {
            'n_estimators': {'type': 'int', 'low': 100, 'high': 1000},
            'learning_rate': {'type': 'float', 'low': 0.01, 'high': 0.3, 'log': False},
            'num_leaves': {'type': 'int', 'low': 20, 'high': 300},
            'max_depth': {'type': 'int', 'low': 3, 'high': 12},
            'min_child_samples': {'type': 'int', 'low': 5, 'high': 100},
            'subsample': {'type': 'float', 'low': 0.5, 'high': 1.0, 'log': False},
            'colsample_bytree': {'type': 'float', 'low': 0.5, 'high': 1.0, 'log': False},
            'reg_alpha': {'type': 'float', 'low': 1e-8, 'high': 10.0, 'log': True},
            'reg_lambda': {'type': 'float', 'low': 1e-8, 'high': 10.0, 'log': True},
        }
        
        # 如果提供了自定义参数空间，则更新默认空间
        self.param_space = self.default_param_space.copy()
        if param_space is not None:
            self.param_space.update(param_space)
    
    def _get_param_suggestions(self, trial: optuna.Trial) -> Dict[str, Any]:
        """
        根据参数空间配置生成Optuna参数建议
        """
        params = {}
        
        for param_name, param_config in self.param_space.items():
            param_type = param_config['type']
            
            if param_type == 'int':
                params[param_name] = trial.suggest_int(
                    param_name, 
                    param_config['low'], 
                    param_config['high']
                )
            elif param_type == 'float':
                log_scale = param_config.get('log', False)
                params[param_name] = trial.suggest_float(
                    param_name,
                    param_config['low'],
                    param_config['high'],
                    log=log_scale
                )
            elif param_type == 'categorical':
                params[param_name] = trial.suggest_categorical(
                    param_name,
                    param_config['choices']
                )
            elif param_type == 'fixed':
                params[param_name] = param_config['value']
        
        return params
    
    def objective(self, trial: optuna.Trial, X: np.ndarray, y: np.ndarray) -> float:
        """
        Optuna优化目标函数
        """
        # 获取参数建议
        param = self._get_param_suggestions(trial)
        
        # 添加固定参数
        param.update({
            'objective': 'binary',
            'metric': 'binary_logloss',
            'verbosity': -1,
            'boosting_type': 'gbdt',
            'random_state': self.random_state,
        })
        
        if self.enable_cv:
            # 使用交叉验证
            return self._objective_cv(trial, X, y, param)
        else:
            # 使用单次训练验证集
            return self._objective_single(trial, X, y, param)
    
    def _objective_cv(self, trial: optuna.Trial, X: np.ndarray, y: np.ndarray, param: Dict) -> float:
        """
        使用交叉验证的目标函数
        """
        kf = StratifiedKFold(n_splits=self.cv_folds, shuffle=True, 
                            random_state=self.random_state)
        
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
        
        return np.mean(cv_scores)
    
    def _objective_single(self, trial: optuna.Trial, X: np.ndarray, y: np.ndarray, param: Dict) -> float:
        """
        使用单次训练验证集的目标函数
        """
        # 划分训练验证集
        X_train, X_val, y_train, y_val = train_test_split(
            X, y, test_size=0.2, random_state=self.random_state, stratify=y
        )
        
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
        return roc_auc_score(y_val, y_pred)
    
    def fit(self, X: np.ndarray, y: np.ndarray, feature_names: Optional[List[str]] = None) -> 'LightGBMOptunaCV':
        """
        训练模型
        
        Parameters:
        - X: 特征数据
        - y: 标签数据
        - feature_names: 特征名称列表
        """
        print("开始超参数优化...")
        print(f"启用交叉验证: {self.enable_cv}")
        if self.enable_cv:
            print(f"交叉验证折数: {self.cv_folds}")
        
        # 创建Optuna study
        self.study = optuna.create_study(
            study_name='LightGBM_Optuna_Study',
            direction='maximize',
            sampler=optuna.samplers.TPESampler(seed=self.random_state),
            pruner=optuna.pruners.MedianPruner(n_warmup_steps=5),
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
        print(f"最佳验证分数: {self.best_score:.4f}")
        
        # 使用最佳参数训练最终模型
        print("训练最终模型...")
        final_params = self.best_params.copy()
        final_params.update({
            'objective': 'binary',
            'metric': 'binary_logloss',
            'verbosity': 1,  # 最终训练时显示日志
            'random_state': self.random_state
        })
        
        # 使用全部数据训练最终模型
        train_data = lgb.Dataset(X, label=y, feature_name=feature_names)
        self.model = lgb.train(
            final_params,
            train_data,
            num_boost_round=final_params.get('n_estimators', 100)
        )
        
        return self
    
    def predict(self, X: np.ndarray) -> np.ndarray:
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
    
    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """
        预测概率（为了与sklearn接口一致）
        """
        predictions = self.predict(X)
        # 转换为二分类概率格式
        return np.column_stack([1 - predictions, predictions])
    
    def evaluate(self, X_test: np.ndarray, y_test: np.ndarray) -> Dict[str, float]:
        """
        评估模型性能
        
        Parameters:
        - X_test: 测试特征
        - y_test: 测试标签
        
        Returns:
        - 评估指标字典
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
    
    def get_feature_importance(self, importance_type: str = 'split') -> pd.Series:
        """
        获取特征重要性
        
        Parameters:
        - importance_type: 'split' 或 'gain'
        
        Returns:
        - 特征重要性Series
        """
        if self.model is None:
            raise ValueError("模型尚未训练，请先调用fit方法")
            
        importance = self.model.feature_importance(importance_type=importance_type)
        feature_names = self.model.feature_name()
        
        return pd.Series(importance, index=feature_names).sort_values(ascending=False)

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
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    
    # 示例1: 使用默认配置（启用交叉验证）
    print("=== 示例1: 使用交叉验证 ===")
    model_cv = LightGBMOptunaCV(
        n_trials=10, 
        cv_folds=5,
        enable_cv=True,
        random_state=42
    )
    model_cv.fit(X_train, y_train)
    model_cv.evaluate(X_test, y_test)
    
    print("\n" + "="*50 + "\n")
    
    # 示例2: 禁用交叉验证，使用自定义参数空间
    print("=== 示例2: 禁用交叉验证 + 自定义参数空间 ===")
    custom_param_space = {
        'n_estimators': {'type': 'int', 'low': 50, 'high': 500},
        'learning_rate': {'type': 'float', 'low': 0.05, 'high': 0.2, 'log': False},
        'num_leaves': {'type': 'int', 'low': 31, 'high': 127},
        'max_depth': {'type': 'fixed', 'value': 8},  # 固定参数
        'min_child_samples': {'type': 'int', 'low': 10, 'high': 50},
        'boosting_type': {'type': 'categorical', 'choices': ['gbdt', 'dart']},  # 新增分类参数
    }
    
    model_no_cv = LightGBMOptunaCV(
        n_trials=10,
        enable_cv=False,  # 禁用交叉验证
        param_space=custom_param_space,
        random_state=42
    )
    model_no_cv.fit(X_train, y_train)
    model_no_cv.evaluate(X_test, y_test)
    
    # 可视化Optuna优化过程
    try:
        import matplotlib.pyplot as plt
        
        # 优化历史
        fig = optuna.visualization.plot_optimization_history(model_cv.study)
        fig.show()
        
        # 超参数重要性
        fig = optuna.visualization.plot_param_importances(model_cv.study)
        fig.show()
        
    except ImportError:
        print("要可视化需要安装plotly: pip install plotly")

if __name__ == "__main__":
    main()
