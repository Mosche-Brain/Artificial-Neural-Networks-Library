set xlabel "x"
set ylabel "y"
set grid
set title "NN sine approximation"

plot "sine_approximation_results.txt" using 1:2 lt 1 lc "green" lw 1 title "targeted values" with lines, \
     "sine_approximation_results.txt" using 1:5 lt 1 lc "blue" lw 1 title "aftertrain approximation (model A)" with lines, \
     "sine_approximation_results.txt" using 1:3 lt 1 lc "red" lw 1 title "pretrain approximation (model A)" with lines, \
     "sine_approximation_results.txt" using 1:4 lt 1 lc "orange" lw 1 title "pretrain approximation (model B)" with lines, \
     "sine_approximation_results.txt" using 1:6 lt 1 lc "violet" lw 1 title "aftertrain approximation (model B)" with lines, \
     "sine_approximation_training_data.txt" using 1:2 lt 1 lc "black" lw 1 title "trainig data"




# "sine_approximation_results.txt" using 1:5 lt 1 lc "blue" lw 1 title "aftertrain approximation (model A)" with lines, \
# "sine_approximation_results.txt" using 1:3 lt 1 lc "red" lw 1 title "pretrain approximation (model A)" with lines, \
