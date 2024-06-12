/**
 * @typedef {Object} Links
 * @prop {string} github Your github repository
 */

/**
 * @typedef {Object} MetaConfig
 * @prop {string} title Your website title
 * @prop {string} description Your website description
 * @prop {string} author Maybe your name
 * @prop {string} siteUrl Your website URL
 * @prop {string} lang Your website Language
 * @prop {string} utterances Github repository to store comments
 * @prop {Links} links
 * @prop {string} favicon Favicon Path
 */

/** @type {MetaConfig} */
const metaConfig = {
  title: "Machine Learning Tutorials",
  description: `A website to host various deep learning and machine learning tutorials`,
  author: "Jordan",
  siteUrl: "https://ml-tutorials.netlify.app",
  lang: "en",
  utterances: "jordandeklerk/machine-learning-tutorials",
  links: {
    github: "https://github.com/jordandeklerk",
  },
  favicon: "src/images/uiuc.png",
}

// eslint-disable-next-line no-undef
module.exports = metaConfig
