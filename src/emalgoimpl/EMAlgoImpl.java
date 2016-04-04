/*
 * To change this license header, choose License Headers in Project Properties.
 * To change this template file, choose Tools | Templates
 * and open the template in the editor.
 */
package emalgoimpl;

import java.io.BufferedReader;
import java.io.FileReader;
import java.io.IOException;
import java.util.ArrayList;
import java.util.Collections;
import java.util.Random;

/**
 *
 * @author Ashish
 */
public class EMAlgoImpl {

    public EMAlgoImpl(int k, String f) {
        this.k = k;
        this.pr = 1.0/this.k;
        this.o = 0.0;
        this.values  = new ArrayList<Double>(6000);
        this.pxb = new ArrayList<>();
        this.pbx = new ArrayList<ArrayList<Double>>(this.k);
        this.sgmList = new ArrayList<Double>(Collections.nCopies(this.k, 0.0));
        this.prior = new ArrayList<Double>(Collections.nCopies(this.k, pr));
        this.bDenom = new ArrayList<Double>(Collections.nCopies(6000, 0.0));
        this.uList = new ArrayList<Double>(Collections.nCopies(this.k, 0.0));
        this.filePath = f;
    }

    Integer k;
    Double pr,o;
    String filePath;
    ArrayList<Double> values;
    ArrayList<ArrayList<Double>> pxb;//initialize all p(x|b) to 0 for each k while defining pxb
    ArrayList<ArrayList<Double>> pbx;//initialize all p(b|x) to 0 for each k
    ArrayList<Double> uList;
    ArrayList<Double> sgmList;
    ArrayList<Double> prior;   //priors pa,pb...all initialised to uniform priors
    ArrayList<Double> bDenom;
    
    public void setO(double o) {
        this.o = o;
    }
    
    public void setO(String o) {
        this.o = calculateInitialVariance();
    }

    
    

    public void setPxb() {
        ArrayList<Double> a = new ArrayList<Double>(Collections.nCopies(6000, 0.0));
        pxb.add(a);
    }

    public void setPbx() {
        ArrayList<Double> a = new ArrayList<Double>(Collections.nCopies(6000, 0.0));
        pbx.add(a);
        
    }
    
    
    /**
     * @param args the command line arguments
     * @throws java.io.IOException
     */
    public static void main(String[] args) throws IOException {
        // TODO code application logic here
        EMAlgoImpl E = new EMAlgoImpl(Integer.parseInt(args[0]),args[2]);
        E.readValuesFromFile();
        if(!args[1].equals("N"))
            E.setO(Double.parseDouble(args[1]));
        else
            E.setO(args[1]);
        E.initEM();
        
    }
    
    void readValuesFromFile() throws IOException{
        try (BufferedReader br = new BufferedReader(new FileReader(filePath))) {
            String line = br.readLine();
            while (line != null) {
                values.add(Double.parseDouble(line));
                line = br.readLine();
            }
        }   
    }
    
    void initEM(){
        Double u[] = new Double[k];
        Double var[] = new Double[k];
        for(int i = 0; i<k;i++){
            u[i] = values.get(new Random().nextInt(values.size()));  //randomly select k means from given set of data
            var[i]=(i+1)*o; //assign initial variance values as o,2o,3o...as multiples of original variance,o
        }
        for(int i = 0;i<k;i++){
            uList.set(i,u[i]);
            sgmList.set(i,var[i]);
        }
        Double previousMean = 0.0;
        int count = 0;
        Boolean converge = true;
        while(converge){
            for(int i=0;i<k;i++){
                setPxb();
                calculatepxuk(uList.get(i),sgmList.get(i),i);
            }
        for(int j=0;j<6000;j++)
                bDenom.set(j, calculateDenomArr(j));
            for(int i=0;i<k;i++){
                setPbx();
                calculatebi(i);
            }
            for(int i=0;i<k;i++){
                calculateMean(i);
            }
            for(int i=0;i<k;i++){
                calculateSigma(i);
            }
            for(int i=0;i<k;i++){
                calculatePrior(i);
            }
            pbx.clear();
            pxb.clear();
            if(previousMean/uList.get(0)>0.9999999 && previousMean/uList.get(0)<1.000000001){
                converge = false;
            }
            previousMean = uList.get(0);
            count++;
        }
        System.out.println("For initial k = "+k+" and initial sigma = "+o+":");
        System.out.println("Final means after "+count+" iterations are: "+uList);
        System.out.println("Final variance after "+count+" iterations are: "+sgmList);
    }

    private double calculateInitialVariance() {
        double sum = 0;
        for(double i : values)
            sum+=i;
        double mean = sum/6000;
        double o = 0;
        for(double j : values)
            o+=(j-mean)*(j-mean);
        o = o/6000;
        return o;//ie o^2/6k
    }

    private void calculatepxuk(Double u, Double var, int m) {
        
        ArrayList<Double> ar = (ArrayList<Double>)pxb.get(m);
        int arIndex = 0;
        for(double i : values){
            double num = calculateExp(i,u,var);
            double den = calculateRoot(var);
            ar.set(arIndex, num/den);
            arIndex++;
        }
        
        pxb.set(m, ar);
        
        
    }

    private double calculateExp(double i, Double u, Double var) {
        double sum = -(i-u)*(i-u);
        sum = sum/(2*var);//since already in o^2 format;no need to square
        return Math.exp(sum);
    }

    private double calculateRoot(Double var) {
        return Math.sqrt(2*Math.PI*var);
    }

    private void calculatebi(int i) {
        ArrayList<Double> ar = (ArrayList<Double>)pxb.get(i);
        ArrayList<Double> bi = (ArrayList<Double>)pbx.get(i);
        for(int l = 0;l<6000;l++){
            double sum = 0;
            sum = bi.get(l);
            sum+=ar.get(l)*prior.get(i)/bDenom.get(l);
            bi.set(l, sum);
        }    
        pbx.set(i, bi);
    }

    private double calculateDenomArr(int j) {
        double sum = 0;
        int i = 0;
        for (ArrayList<Double> ar1 : pxb) {
            sum = sum + ar1.get(j)*prior.get(i);
            i++;
        }
        
        return sum;
    }

    private void calculateMean(int i) {
        double num = 0, den = 0;
        ArrayList<Double> bi = (ArrayList<Double>)pbx.get(i);
        for(int j = 0;j<6000;j++){
            num=num+bi.get(j)*values.get(j);
            den+=bi.get(j);
        }
        uList.set(i, num/den);
    }

    private void calculateSigma(int i) {
        double num = 0, den = 0;
        ArrayList<Double> bi = (ArrayList<Double>)pbx.get(i);
        for(int j = 0;j<6000;j++){
            num=num+bi.get(j)*(values.get(j)-uList.get(i))*(values.get(j)-uList.get(i));
            den+=bi.get(j);
        }
        sgmList.set(i, num/den);
    }

    private void calculatePrior(int i) {
        double sum = 0;
        ArrayList<Double> bi = (ArrayList<Double>)pbx.get(i);
        for(int j = 0;j<6000;j++){
            sum+=bi.get(j);
        }
        prior.set(i, sum/6000);
    }
    
}
