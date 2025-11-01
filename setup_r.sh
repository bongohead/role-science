# Install R & Rstudio server
mkdir -p /etc/apt/keyrings
wget -qO- https://cloud.r-project.org/bin/linux/ubuntu/marutter_pubkey.asc \
  | tee /etc/apt/keyrings/cran.asc

echo "deb [signed-by=/etc/apt/keyrings/cran.asc] \
https://cloud.r-project.org/bin/linux/ubuntu $(lsb_release -cs)-cran40/" \
 | tee /etc/apt/sources.list.d/cran.list

apt update -qq
apt install -y --no-install-recommends r-base r-base-dev

# Set locales
apt-get install -y locales
sed -i 's/^# *en_US.UTF-8 UTF-8/en_US.UTF-8 UTF-8/' /etc/locale.gen
locale-gen
update-locale LANG=en_US.UTF-8

# Install kernel
R -e "install.packages('IRkernel', Ncpus = 8); IRkernel::installspec(user = FALSE)"

jupyter kernelspec list

apt-get install -y libxml2-dev libfontconfig1-dev libcurl4-openssl-dev libharfbuzz-dev libfribidi-dev libfreetype6-dev libpng-dev libtiff5-dev libjpeg-dev libwebp-dev libssl-dev
R -e "install.packages('tidyverse', Ncpus = 8);"
R -e "install.packages('patchwork', Ncpus = 8);"
R -e "install.packages('slider', Ncpus = 8);"
R -e "install.packages('zoo', Ncpus = 8);"
R -e "install.packages('arrow', Ncpus = 8);"

# Visualizations for paths
# apt get install -y pandoc
# R -e "install.packages('reticulate', Ncpus = 8);"
# R -e "install.packages('highcharter', Ncpus = 8);"

# Fonts for plotting
apt install -y fonts-texgyre
R -e "install.packages('showtext', Ncpus = 8);"
R -e "install.packages('svglite', Ncpus = 8);"
R -e "install.packages('ggtext', Ncpus = 8);"

# Fix issue with list-cols not displaying
R -e "install.packages('remotes'); remotes::install_github('IRkernel/repr'); IRkernel::installspec()"