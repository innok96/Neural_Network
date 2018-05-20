package com.company;

import com.company.Layers.BackpropLayer;
import com.company.Layers.SigmoidLayer;

import javax.imageio.ImageIO;
import javax.swing.*;
import java.awt.image.BufferedImage;
import java.io.File;
import java.lang.String;

import static java.lang.Double.max;

public class Main {
    static BackpropNetwork network;
    static double[] input = new double[48*48*3];
    static double totalError=100;
    static double[] answer;
    static int accuracy=0;

    private static void getInput(BufferedImage imag){
        for(int i=0, ii=0; i<48; i++){
            for(int j=0; j<48; j++){
                RGB rgb = new RGB(imag.getRGB(i, j));
                input[ii++]=rgb.R/255; input[ii++]=rgb.G/255; input[ii++]=rgb.B/255;
            }
        }
    }

    private static void learn(String path, double[] goal) throws Exception{
        for(int ind=1; ind<=20; ind++){
            BufferedImage imag = ImageIO.read(new File("D:\\projects\\Neural_Network\\src\\labels\\"+path+"\\"+ind+".jpg"));
            applySobel(imag);
            getInput(imag);
            totalError=max(network.learnPattern(input, goal, 0.1, 0.1), totalError);
        }
    }

    private static void check(String path) throws Exception{
        for(int ind=21; ind<=25; ind++){
            BufferedImage imag = ImageIO.read(new File("D:\\projects\\Neural_Network\\src\\labels\\"+path+"\\"+ind+".jpg"));
            getInput(imag);
            answer = network.computeOutput(input);
            BufferedImage img = ImageIO.read(new File("D:\\projects\\Neural_Network\\src\\images\\"+path+"\\"+ind+".jpg"));
            GUI gui= new GUI(path, answer, img);
            gui.setVisible(true);
            String ans=null;
            if(answer[0]>answer[1] && answer[0]>answer[2]) ans="Деревья";
            if(answer[1]>answer[0] && answer[1]>answer[2]) ans="Лебеди";
            if(answer[2]>answer[1] && answer[2]>answer[0]) ans="Облака";
            if(ans.equals(path)) accuracy++;
        }
    }

    private static void applySobel(BufferedImage imag){
        int N = 3;
        double[][] d1 = {{-1, -2, -1}, {0, 0, 0}, {1, 2, 1}},
                d2 = {{-1, 0, 1}, {-2, 0, 2}, {-1, 0, 1}};
        double pixR1[][] = new double[imag.getWidth()][imag.getHeight()];
        double pixG1[][] = new double[imag.getWidth()][imag.getHeight()];
        double pixB1[][] = new double[imag.getWidth()][imag.getHeight()];
        double pixR2[][] = new double[imag.getWidth()][imag.getHeight()];
        double pixG2[][] = new double[imag.getWidth()][imag.getHeight()];
        double pixB2[][] = new double[imag.getWidth()][imag.getHeight()];
        for (int i = 0; i < imag.getWidth(); i++) {
            for (int j = 0; j < imag.getHeight(); j++) {
                for (int ii = i - N / 2; ii <= i + N / 2; ii++) {
                    for (int jj = j - N / 2; jj <= j + N / 2; jj++) {
                        RGB rgb;
                        if (ii >= 0 && ii < imag.getWidth() && jj >= 0 && jj < imag.getHeight())
                            rgb = new RGB(imag.getRGB(ii, j));
                        else {
                            int ii1 = ii, jj1 = jj;
                            if (ii1 < 0) ii1 = 0;
                            if (ii1 >= imag.getWidth()) ii1 = imag.getWidth() - 1;
                            if (jj1 < 0) jj1 = 0;
                            if (jj1 >= imag.getHeight()) jj1 = imag.getHeight() - 1;
                            rgb = new RGB(imag.getRGB(ii1, jj1));
                        }
                        pixR1[i][j] += rgb.R * d1[ii - i + N / 2][jj - j + N / 2];
                        pixG1[i][j] += rgb.G * d1[ii - i + N / 2][jj - j + N / 2];
                        pixB1[i][j] += rgb.B * d1[ii - i + N / 2][jj - j + N / 2];
                        pixR2[i][j] += rgb.R * d2[ii - i + N / 2][jj - j + N / 2];
                        pixG2[i][j] += rgb.G * d2[ii - i + N / 2][jj - j + N / 2];
                        pixB2[i][j] += rgb.B * d2[ii - i + N / 2][jj - j + N / 2];
                    }
                }
            }
        }
        for (int i = 0; i < pixR1.length; i++) {
            for (int j = 0; j < pixR1[i].length; j++) {
                RGB rgb = new RGB((int) Math.sqrt(pixR1[i][j] * pixR1[i][j] + pixR2[i][j] * pixR2[i][j]),
                        (int) Math.sqrt(pixG1[i][j] * pixG1[i][j] + pixG2[i][j] * pixG2[i][j]),
                        (int) Math.sqrt(pixB1[i][j] * pixB1[i][j] + pixB2[i][j] * pixB2[i][j]));
                imag.setRGB(i, j, rgb.toInt((imag.getRGB(i, j) >> 24) & 0xff));
            }
        }
    }

    public static void main(String[] args) throws Exception{
        String networkPath = "D:\\projects\\Neural_Network\\src\\com\\company\\1";
        try {
            network = (BackpropNetwork) Network.loadFromFile(networkPath);
        } catch(Exception ex){}
        while(totalError>1){
            totalError=0;
            double[] goal = new double[3];
            goal[0]=1;
            learn("Деревья", goal);
            goal[0]=0; goal[1]=1;
            learn("Лебеди", goal);
            goal[1]=0; goal[2]=1;
            learn("Облака", goal);
        }
        check("Деревья");
        check("Лебеди");
        check("Облака");
        System.out.println("Accuracy = " + accuracy/15.);
    }
}
