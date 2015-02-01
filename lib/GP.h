#ifndef _GP_H_
#define _GP_H_


/// --- This is a piece of somehow old code we want to compare against

#include <Eigen/Dense>
#include <vector>
#include <cstdio>
#include <iostream>

// overwrites A
static void shermanMorrisonUpdate( Eigen::MatrixXd &Ainv, const Eigen::VectorXd &v, const Eigen::VectorXd &u )
{
    double k = 1.0 / (1.0 + (v.transpose() * Ainv * u)(0) );
    Ainv -= k * Ainv * u * v.transpose() * Ainv;
}

static void gpInvUpdate( Eigen::MatrixXd &Ainv, const Eigen::VectorXd &k )
{
    const unsigned N = Ainv.cols();
    Eigen::VectorXd v(N);
    v.setZero();
    v.coeffRef( N-1 ) = 1.0;

    Eigen::VectorXd kp = k;
    kp.coeffRef(N-1) = 0;

    shermanMorrisonUpdate( Ainv, v, kp );

    kp.coeffRef(N-1) = k.coeffRef(N-1) - 1.0;

    shermanMorrisonUpdate( Ainv, kp, v );
}

class GP
{
public:
    typedef Eigen::VectorXd VectorType;
    typedef Eigen::MatrixXd MatrixType;
            
protected:
    VectorType  mX, mY, mPtMeasNoise;    // observation, x and y
    VectorType  mYExtra;    // only for debugging
    MatrixType  mInvK; // inverse kernel matrix
    double      mMeasNoise; // sigma^2 of measurement noise

    VectorType  mCachedInvKxY;

    unsigned mTotalNumSamples;

    inline double meanFunc( double x ) const
    {
        return 0;
    }
    
    inline double kernelFunc( double x, double y ) const
    {
        //return 100 + 100*( 1.0*exp( -0.5 * (x-y)*(x-y) * (1.0 / 10.0) ) );
        return 100 + 100*( 1.0*exp( -0.5 * (x-y)*(x-y) * (1.0 / 10.0) ) );
    }
    
public:

    void setTotalNumberOfSamples( unsigned N )
    {
        mTotalNumSamples = N;
    }

    void init()
    {
        mX.resize(0);
        mY.resize(0);
        mPtMeasNoise.resize(0,0);
        mInvK.resize(0,0);
        
        mMeasNoise = 1.0;
    }

    GP()
    {
        init();
    }

    void cacheUpdate()
    {
        VectorType meanVals( mX.size() );  //this can be optimized!
        for (unsigned i=0; i < mX.size(); i++)
            meanVals.coeffRef(i) = meanFunc( mX.coeff(i) );

        mCachedInvKxY = mInvK * (mY - meanVals);
    }

    void updateKnownVariance( double x, double y, double measVar, double yExtra )
    {
        // 'future' size
        const unsigned N = mX.size() + 1;
        
        mX.conservativeResize( N );
        mY.conservativeResize( N );
        mYExtra.conservativeResize( N );
        mPtMeasNoise.conservativeResize( N );
        
        mInvK.conservativeResize( N, N );
        mInvK.row(N-1).setZero();
        mInvK.col(N-1).setZero();
        mInvK.coeffRef(N-1,N-1) = 1.0;

        mX.coeffRef(N-1) = x;
        mY.coeffRef(N-1) = y;
        mYExtra.coeffRef(N-1) = yExtra;

        // meas noise, scaled by number of samples
        const double mNoise = measVar;


        VectorType k(N);
        for (unsigned i=0; i < N; i++)
            k.coeffRef(i) = kernelFunc( x, mX.coeff(i) );

        k.coeffRef(N-1) += mNoise;

        // update noise covariance
        for (unsigned i=0; i < N-1; i++)
        {
            //const double ccMult = 0.99 * exp(-0.5*pow( mX.coeff(i) - x, 2) / 0.01 ) ;
            //const double vv = ccMult * std::min(mPtMeasNoise.coeff(i), mNoise) ;
            
            const double vv = 0.0;
            //const double ccMult = 0.1;
            //const double vv = ccMult * sqrt( mPtMeasNoise.coeff(i) * mNoise );
            k.coeffRef(i) += vv;
        }

        gpInvUpdate( mInvK, k );
        
        mPtMeasNoise.coeffRef(N-1) = mNoise;
        cacheUpdate();
    }
    
    // when measurement comes
    //  yExtra is just for debugging
    void update( double x, double y, unsigned numSamp, double yExtra )
    {
        // meas noise, scaled by number of samples
        const double mNoise = ( mTotalNumSamples * mMeasNoise ) / numSamp;

        updateKnownVariance( x, y, mNoise, yExtra );
    }
    
    double predict( double x, double *var = 0 )
    {
        const unsigned N = mX.size();

        if (N == 0)
        {
            // just return some high variance
            *var = 1000;
            return x;
        }
        
        VectorType k(N);
        for (unsigned i=0; i < N; i++)
            k.coeffRef(i) = kernelFunc( x, mX.coeff(i) );
        
        const double m = meanFunc(x) + k.transpose() * mCachedInvKxY;
        
        if (var != 0)
        {
            const double c = kernelFunc(x, x);// + mMeasNoise;
            *var = c - k.transpose() * mInvK * k;
        }
        
        return m;
    }

public:

    void printInfo() const
    {
        std::cout << "mX: " << std::endl << mX << std::endl;
        std::cout << "mY: " << std::endl << mY << std::endl;
        std::cout << "mYExtra: " << std::endl << mYExtra << std::endl;
        std::cout << "mPtMeasNoise: " << std::endl << mPtMeasNoise << std::endl;
        std::cout << "mInvK: " << std::endl << mInvK << std::endl;
    }
};


typedef std::vector<GP>   GPList;

#endif
