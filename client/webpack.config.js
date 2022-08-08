const path = require('path');
 
module.exports = {
  entry: path.resolve(__dirname, './src/adatest.jsx'),
  module: {
    rules: [
      {
        test: /\.(js|jsx)$/,
        exclude: /node_modules/,
        use: ['babel-loader'],
      },
      {
        test: /\.css$/i,
        use: ["style-loader", "css-loader"],
      },
      {
        test: /\.(png|woff|woff2|eot|ttf|svg)$/,
        use: ['url-loader'],
      },
      { // this allows font-awesome to be used during development mode... (since we print to the page in a script tag)
        test: /\.js$/,
        loader: 'string-replace-loader',
        options: {
          search: '</script>',
          replace: '_/script>',
        }
      }
    ],
  },
  resolve: {
    extensions: ['*', '.js', '.jsx'],
  },
  externals: {
    // 'react': 'React',
    // 'react-dom': 'ReactDOM'
  },
  output: {
    path: path.resolve(__dirname, '../adatest/resources'),
    filename: 'main.js',
  },
  mode: "production"
  // mode: "development"
};