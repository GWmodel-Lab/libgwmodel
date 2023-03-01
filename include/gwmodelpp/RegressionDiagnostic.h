#ifndef REGRESSIONDIAGNOSTIC_H
#define REGRESSIONDIAGNOSTIC_H

namespace gwm
{

/**
 * @brief \~english Diagnostic information of regression model. \~chinese 回归模型诊断信息。
 * 
 */
struct RegressionDiagnostic
{
    double RSS;             //!< \~english Sum of squared residual \~chinese 残差平方和
    double AIC;             //!< \~english Akaike information criterion  \~chinese 赤池信息准则指数
    double AICc;            //!< \~english Corrected Akaike information criterion \~chinese 修正赤池信息准则指数
    double ENP;             //!< \~english Effective number of parameters \~chinese 有效参数数
    double EDF;             //!< \~english Effective degrees of freedom \~chinese 有效自由度
    double RSquare;         //!< \f$R^2\f$
    double RSquareAdjust;   //!< \~english Adjusted \f$R^2\f$ \~chinese 调整 \f$R^2\f$ 值
};

}

#endif  // REGRESSIONDIAGNOSTIC_H