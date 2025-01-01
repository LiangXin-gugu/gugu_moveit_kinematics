#include <gugu_ur10e_analytic_kinematics_plugin/ur_kin_gugu.h>

#include <math.h>
#include <stdio.h>


namespace ur_kinematics {

  namespace {
    const double ZERO_THRESH = 0.00000001;
    int SIGN(double x) {
      return (x > 0) - (x < 0);
    }
    const double PI = M_PI;

    #define UR10e_PARAMS
    #ifdef UR10e_PARAMS
    const double d1 =  0.1807;
    const double a2 = -0.6127;
    const double a3 = -0.57155;
    const double d4 =  0.17415;
    const double d5 =  0.11985;
    const double d6 =  0.11655;
    #endif
  }

  int inverse_gugu(const double* T, double* q_sols, double q6_des) {
    int num_sols = 0;
    double T00 =  *T; T++; double T01 =  *T; T++; double T02 =  *T; T++; double T03 =  *T; T++; 
    double T10 =  *T; T++; double T11 =  *T; T++; double T12 =  *T; T++; double T13 =  *T; T++; 
    double T20 =  *T; T++; double T21 =  *T; T++; double T22 =  *T; T++; double T23 =  *T;

    ////////////////////////////// shoulder rotate joint (q1) //////////////////////////////
    double q1[2];
    {
      double m = T03 - d6*T02;
      double n = -T13 + d6*T12;
      double rou_square = m*m + n*n;
      double rou = sqrt(rou_square);

      if (d4 > rou) {
        return num_sols;
      }
      else {
        double numer = d4/rou;
        double div = sqrt(1-d4*d4/rou_square);
        q1[0] = atan2(numer, div) - atan2(n, m);
        q1[1] = atan2(numer, -div) - atan2(n, m);
      }
    }
    ////////////////////////////////////////////////////////////////////////////////

    ////////////////////////////// wrist 2 joint (q5) //////////////////////////////
    double q5[2][2];
    {
      for(int i=0;i<2;i++) {
        double c1 = cos(q1[i]), s1 = sin(q1[i]);        
        double numer = (-d4 + s1*T03 - c1*T13);
        double div;
        if(fabs(fabs(numer) - fabs(d6)) < ZERO_THRESH)
          div = SIGN(numer) * SIGN(d6);
        else
          div = numer / d6;
        double arccos = acos(div);
        q5[i][0] = arccos;
        q5[i][1] = -arccos;
      }
    }
    ////////////////////////////////////////////////////////////////////////////////

    {
      for(int i=0;i<2;i++) {
        for(int j=0;j<2;j++) {
          double c1 = cos(q1[i]), s1 = sin(q1[i]);
          double c5 = cos(q5[i][j]), s5 = sin(q5[i][j]);          
          double q6;
          ////////////////////////////// wrist 3 joint (q6) //////////////////////////////
          if(fabs(s5) < ZERO_THRESH)
            q6 = q6_des;
          else {
            q6 = atan2(SIGN(s5)*-(T01*s1 - T11*c1), 
                       SIGN(s5)*(T00*s1 - T10*c1));
            if(fabs(q6) < ZERO_THRESH)
              q6 = 0.0;
          }
          ////////////////////////////////////////////////////////////////////////////////

          double q2[2], q3[2], q4[2];
          ///////////////////////////// RRR joints (q2,q3,q4) ////////////////////////////
          double c6 = cos(q6), s6 = sin(q6);
          double u = d5*(s6*(T00*c1 + T10*s1) + c6*(T01*c1 + T11*s1)) - d6*(T02*c1 + T12*s1) + 
                        T03*c1 + T13*s1;
          double w = d5*(T21*c6 + T20*s6) - d6*T22 + T23 - d1;
          double k11 = c5*(c6*(T00*c1 + T10*s1) - s6*(T01*c1 + T11*s1)) - s5*(T02*c1 + T12*s1);
          double k31 = c5*(T20*c6 - T21*s6) - T22*s5;

          double costheta3 = (u*u + w*w - a2*a2 - a3*a3) / (2.0*a2*a3);
          if(fabs(fabs(costheta3) - 1.0) < ZERO_THRESH)
            costheta3 = SIGN(costheta3);
          else if(fabs(costheta3) > 1.0) {
            // TODO NO SOLUTION
            continue;
          }
          double arccos = acos(costheta3);
          q3[0] = arccos;
          q3[1] = -arccos;
          for(int k=0;k<2;k++) {
            double s3 = sin(q3[k]), c3 = cos(q3[k]);
            double numer = a3*s3;
            double div = a2 + a3*c3;
            q2[k] = atan2(w, u) - atan2(numer, div);
            if (q2[k] < -PI) {
              q2[k] += 2*PI;
            }
            else if (q2[k] > PI) {
              q2[k] -= 2*PI;
            }
            q4[k] = atan2(k31, k11) - q2[k] - q3[k];
            if (q4[k] < -PI) {
              q4[k] += 2*PI;
            }
            else if (q4[k] > PI) {
              q4[k] -= 2*PI;
            }

            q_sols[num_sols*6+0] = q1[i];    q_sols[num_sols*6+1] = q2[k]; 
            q_sols[num_sols*6+2] = q3[k];    q_sols[num_sols*6+3] = q4[k]; 
            q_sols[num_sols*6+4] = q5[i][j]; q_sols[num_sols*6+5] = q6; 
            num_sols++;
          }
        }
      }
    }
    
    // //gugu add: limit sols scope in -pi to pi.
    // for (int i=0;i<num_sols;i++){
    //   for (int j=0;j<6;j++){
    //     if (q_sols[i*6+j] < -PI){
    //       q_sols[i*6+j] = q_sols[i*6+j] + 2*PI;
    //     }
    //     else if (q_sols[i*6+j] > PI){
    //       q_sols[i*6+j] = q_sols[i*6+j] - 2*PI;
    //     }
    //   }
    // }
    
    return num_sols;
  }
};
