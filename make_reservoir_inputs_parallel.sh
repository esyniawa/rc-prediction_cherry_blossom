mean_pop=("5" "10" "20" "30" "60" "120")

for mean in "${mean_pop[@]}"; do
  python input_dynamic_data.py 1 $mean &
done