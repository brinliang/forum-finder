/* eslint-disable */
const path = require('path')
const TerserPlugin = require("terser-webpack-plugin");

module.exports = {
  entry: './src/index.tsx',
  module: {
    rules: [
      {
        test: /\.tsx?$/,
        use: 'ts-loader',
        exclude: /node_modules/,
      },
    ],
  },
  resolve: {
    extensions: ['.tsx', '.ts', '.js'],
  },
  output: {
    filename: 'main.js',
    path: path.resolve(__dirname, '../build'),
  },
  devServer: {
    static: path.resolve(__dirname, '.'),
    compress: true,
    port: 3000,
  },
  devtool: 'source-map',
  optimization: {
    minimize: true,
    minimizer: [new TerserPlugin()],
  },
};