const path = require('path');

const isDevelopment = process.env.NODE_ENV === 'dev';
const ASSET_PATH = process.env.ASSET_PATH || '/';

module.exports = {
  entry: path.resolve(__dirname, './src/adatest.jsx'),
  devtool: isDevelopment ? 'eval-source-map' : false,
  module: {
    rules: [
      {
        test: /\.(js|jsx)$/,
        exclude: /node_modules/,
        use: ['babel-loader'],
      },
      {
        test: /\.ts(x?)$/,
        exclude: /node_modules/,
        use: [
          {
              loader: "ts-loader"
          }
        ]
      },
      {
        test: /\.css$/i,
        use: ["style-loader", "css-loader"],
      },
      { // this allows font-awesome to be used during development mode... (since we print to the page in a script tag)
        test: /\.js$/,
        loader: 'string-replace-loader',
        options: {
          search: '</script>',
          replace: '_/script>',
        }
      },
      // https://github.com/webpack-contrib/source-map-loader
      {
        enforce: "pre",
        test: /\.js$/,
        loader: "source-map-loader"
      },
      {
        test: /\.(png|jpe?g|gif)$/i,
        use: [
          {
            loader: 'file-loader',
          },
        ],
      }
    ],
  },
  resolve: {
    extensions: ['*', '.js', '.jsx', '.ts', '.tsx'],
  },
  externals: {
    // 'react': 'React',
    // 'react-dom': 'ReactDOM'
  },
  output: {
    path: path.resolve(__dirname, '../adatest/resources'),
    publicPath: ASSET_PATH,
    filename: 'main.js',
  },
  mode: isDevelopment ? "development" : "production"
};