% MetaOpticsAI — bundled photonics literature corpus.
% Each entry corresponds to a PDF in ../papers/ and is referenced by
% the Self-RAG knowledge base (core/automl.MetalensKnowledgeBase).

@article{moharam_gaylord_1981,
  title   = {Rigorous coupled-wave analysis of planar-grating diffraction},
  author  = {Moharam, M. G. and Gaylord, T. K.},
  journal = {Journal of the Optical Society of America},
  volume  = {71},
  number  = {7},
  pages   = {811--818},
  year    = {1981},
  doi     = {10.1364/JOSA.71.000811},
  note    = {Foundational RCWA paper; describes the eigenvalue formulation
             implemented in core/rcwa_engine.py}
}

@article{li_1996,
  title   = {Use of Fourier series in the analysis of discontinuous periodic structures},
  author  = {Li, Lifeng},
  journal = {Journal of the Optical Society of America A},
  volume  = {13},
  number  = {9},
  pages   = {1870--1876},
  year    = {1996},
  doi     = {10.1364/JOSAA.13.001870},
  note    = {Correct Fourier-factorisation rules for fast RCWA convergence}
}

@article{khorasaninejad_2016,
  title   = {Metalenses at visible wavelengths: Diffraction-limited focusing
             and subwavelength resolution imaging},
  author  = {Khorasaninejad, Mohammadreza and Chen, Wei Ting and Devlin,
             Robert C. and Oh, Jaewon and Zhu, Alexander Y. and Capasso,
             Federico},
  journal = {Science},
  volume  = {352},
  number  = {6290},
  pages   = {1190--1194},
  year    = {2016},
  doi     = {10.1126/science.aaf6644},
  note    = {Canonical TiO2 metalens demonstration, design parameters
             reused in our 532 nm preset}
}

@article{yoon_rho_2021,
  title   = {MAXIM: Metasurfaces-oriented electromagnetic wave simulation
             software with intuitive graphical user interfaces},
  author  = {Yoon, Gwanho and Rho, Junsuk},
  journal = {Computer Physics Communications},
  volume  = {264},
  pages   = {107846},
  year    = {2021},
  doi     = {10.1016/j.cpc.2021.107846},
  note    = {S-matrix formulation followed by core/rcwa_engine.py}
}

@article{hughes_2018,
  title   = {Adjoint Method and Inverse Design for Nonlinear Nanophotonic
             Devices},
  author  = {Hughes, Tyler W. and Minkov, Momchil and Williamson, Ian A. D.
             and Fan, Shanhui},
  journal = {ACS Photonics},
  volume  = {5},
  number  = {12},
  pages   = {4781--4787},
  year    = {2018},
  doi     = {10.1021/acsphotonics.8b01522}
}

@article{jensen_sigmund_2011,
  title   = {Topology optimization for nano-photonics},
  author  = {Jensen, J. S. and Sigmund, O.},
  journal = {Laser \& Photonics Reviews},
  volume  = {5},
  number  = {2},
  pages   = {308--321},
  year    = {2011},
  doi     = {10.1002/lpor.201000014}
}

@article{malkiel_2018,
  title   = {Plasmonic nanostructure design and characterization via Deep
             Learning},
  author  = {Malkiel, Itzik and Mrejen, Michael and Nagler, Achiya and
             Arieli, Uri and Wolf, Lior and Suchowski, Haim},
  journal = {Light: Science \& Applications},
  volume  = {7},
  number  = {60},
  year    = {2018},
  doi     = {10.1038/s41377-018-0060-7}
}

@article{asai_self_rag_2024,
  title   = {Self-{RAG}: Learning to Retrieve, Generate, and Critique
             through Self-Reflection},
  author  = {Asai, Akari and Wu, Zeqiu and Wang, Yizhong and Sil, Avirup
             and Hajishirzi, Hannaneh},
  booktitle = {ICLR},
  year    = {2024},
  note    = {Reflection pattern used in core/automl.py LangGraph pipeline}
}