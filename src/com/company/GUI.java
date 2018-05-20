package com.company;

import javax.swing.*;
import java.awt.image.BufferedImage;

public class GUI extends JFrame {
    public GUI(String title, double[] answer, BufferedImage imag){
        super(title);
        setResizable(false);
        setLayout(null);
        setBounds(100, 100, 800, 600);
        JLabel label = new JLabel();
        label.setIcon(new ImageIcon(imag));
        label.setBounds(0, 0, 800, 500);
        add(label);
        double sum=answer[0]+answer[1]+answer[2];
        String prediction = String.format("Tree: %d %% Swan: %d %% Cloud: %d %%", (int)(answer[0]*100/sum), (int)(answer[1]*100/sum), (int)(answer[2]*100/sum));
        JLabel label1 = new JLabel(prediction);
        label1.setBounds(0, 510, 500, 80);
        add(label1);
    }
}
