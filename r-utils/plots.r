theme_iclr = function(base_size = 22, base_family = "TeX Gyre Termes", grid = T, legend_position = "bottom") {

  # Only import if extrafont database is empty or missing the font
  if (nrow(filter(systemfonts::system_fonts(), str_detect(family, base_family))) == 0) {
    stop('Error, missing desired font font!')
  }

  theme =
    theme_bw(base_size = base_size, base_family = base_family) %+replace%
    theme(
      # Titles
      plot.title = element_blank(), 
      plot.subtitle = element_blank(),
      
      # Axes
      axis.title = ggtext::element_markdown(face = 'plain', size = rel(1.0)), 
      axis.title.x = ggtext::element_markdown(),
      axis.title.y = ggtext::element_markdown(),
      axis.text.x = element_text(angle = 45, hjust = 1, vjust = 1, size = rel(0.9)),
      axis.text.y = element_text(size = rel(0.9)),

      # Legend
      legend.position = legend_position,
      legend.title = element_text(face = 'bold', size = rel(0.9)),
      legend.text = ggtext::element_markdown(size = rel(0.9)),
      legend.background = element_blank(),
      # element_rect(fill = "white", color = "grey90", linewidth = 0.2),
      legend.key.size = unit(0.8, "lines"),
      legend.margin = margin(0, 0, 0, 0),

      # Grid and panel
      panel.grid.major = if(grid) {
        element_line(colour = "grey85", linewidth = 0.3)
      } else {
        element_blank()
      },
      panel.grid.minor = element_blank(),
      panel.border = element_rect(colour = "grey20", fill = NA, linewidth = 0.5),
      # panel.grid.major = element_line(colour = "grey85", linewidth = 0.3),
      # panel.border = element_rect(colour = "grey70", fill = NA, linewidth = 0.5),
      
      # Facets
      strip.background = element_blank(),
      strip.text = element_text(face = "bold", size = rel(1.0))
    )

  return(theme)
}